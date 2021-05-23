// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::{abort, proc_macro_error};
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[cfg(feature = "tex")]
use std::fs;

fn get_bits(expr_call: &syn::ExprCall) -> syn::Expr {
    if let syn::Expr::Path(ep) = &*expr_call.func {
        if !ep.path.is_ident("Bits") {
            abort!(
                expr_call,
                "Unexpected function name in coder: {}",
                ep.path.get_ident().unwrap()
            );
        }
        if expr_call.args.len() != 1 {
            abort!(
                expr_call,
                "Unexpected number of arguments for Bits() in coder: {}",
                expr_call.args.len()
            );
        }
        return expr_call.args[0].clone();
    }
    abort!(expr_call, "Unexpected function call in coder");
}

fn parse_single_coder(input: &syn::Expr, extra_lit: Option<&syn::ExprLit>) -> TokenStream2 {
    match &input {
        syn::Expr::Lit(lit) => match extra_lit {
            None => quote! {U32::Val(#lit)},
            Some(elit) => quote! {U32::Val(#lit + #elit)},
        },
        syn::Expr::Call(expr_call) => {
            let bits = get_bits(&expr_call);
            match extra_lit {
                None => quote! {U32::Bits(#bits)},
                Some(elit) => quote! {U32::BitsOffset{n: #bits, off: #elit}},
            }
        }
        syn::Expr::Binary(syn::ExprBinary {
            attrs: _,
            left,
            op: syn::BinOp::Add(_),
            right,
        }) => {
            let (left, right) = if let syn::Expr::Lit(_) = **left {
                (right, left)
            } else {
                (left, right)
            };
            match (&**left, &**right) {
                (syn::Expr::Call(expr_call), syn::Expr::Lit(lit)) => {
                    let bits = get_bits(&expr_call);
                    match extra_lit {
                        None => quote! {U32::BitsOffset{n: #bits, off: #lit}},
                        Some(elit) => quote! {U32::BitsOffset{n: #bits, off: #lit + #elit}},
                    }
                }
                _ => abort!(
                    input,
                    "Unexpected expression in coder, must be Bits(a) + b, Bits(a), or b"
                ),
            }
        }
        _ => abort!(
            input,
            "Unexpected expression in coder, must be Bits(a) + b, Bits(a), or b"
        ),
    }
}

fn parse_coder(input: syn::Expr) -> TokenStream2 {
    let parse_u2s = |expr_call: &syn::ExprCall, lit: Option<&syn::ExprLit>| {
        if let syn::Expr::Path(ep) = &*expr_call.func {
            if !ep.path.is_ident("u2S") {
                return parse_single_coder(&input, None);
            }
            if expr_call.args.len() != 4 {
                abort!(
                    input,
                    "Unexpected number of arguments for U32() in coder: {}",
                    expr_call.args.len()
                );
            }
            let args = vec![
                parse_single_coder(&expr_call.args[0], lit),
                parse_single_coder(&expr_call.args[1], lit),
                parse_single_coder(&expr_call.args[2], lit),
                parse_single_coder(&expr_call.args[3], lit),
            ];
            return quote! {U32Coder(#(#args),*)};
        }
        abort!(input, "Unexpected function call in coder");
    };

    match &input {
        syn::Expr::Call(expr_call) => parse_u2s(&expr_call, None),
        syn::Expr::Binary(syn::ExprBinary {
            attrs: _,
            left,
            op: syn::BinOp::Add(_),
            right,
        }) => {
            let (left, right) = if let syn::Expr::Lit(_) = **left {
                (right, left)
            } else {
                (left, right)
            };
            match (&**left, &**right) {
                (syn::Expr::Call(expr_call), syn::Expr::Lit(lit)) => {
                    return parse_u2s(&expr_call, Some(&lit));
                }
                _ => abort!(
                    input,
                    "Unexpected expression in coder, must be (u2S|Bits)(a) + b, (u2S|Bits)(a), or b"
                ),
            }
        }
        _ => return parse_single_coder(&input, None),
    }
}

fn is_primitive_type(ty: &syn::Type, sty: &str) -> bool {
    if let syn::Type::Path(syn::TypePath { qself: _, path }) = ty {
        path.is_ident(sty)
    } else {
        false
    }
}

fn prettify_condition(cond: &syn::Expr) -> String {
    (quote! {#cond})
        .to_string()
        .replace(" . ", ".")
        .replace("! ", "!")
}

fn prettify_coder(coder: &syn::Expr) -> String {
    (quote! {#coder}).to_string()
}

#[derive(Debug)]
enum Type {
    Nested(String),
    Bool,
    U32(TokenStream2, String),
}

#[derive(Debug)]
struct Field {
    name: proc_macro2::Ident,
    condition: Option<(syn::Expr, String)>,
    default: TokenStream2,
    ty: Type,
}

impl Field {
    fn parse(f: &syn::Field) -> Field {
        let mut condition = None;
        let mut default = None;
        let mut coder = None;

        // Parse attributes.
        for a in &f.attrs {
            if a.path.is_ident("coder") {
                if coder.is_some() {
                    abort!(f, "Repeated coder");
                }
                let coder_ast = a.parse_args::<syn::Expr>().unwrap();
                let pretty_coder = prettify_coder(&coder_ast);
                coder = Some((parse_coder(coder_ast), pretty_coder));
            } else if a.path.is_ident("default") {
                if default.is_some() {
                    abort!(f, "Repeated default");
                }
                let default_expr = a.parse_args::<syn::Expr>().unwrap();
                default = Some(quote! {#default_expr});
            } else if a.path.is_ident("condition") {
                if condition.is_some() {
                    abort!(f, "Repeated condition");
                }
                let condition_ast = a.parse_args::<syn::Expr>().unwrap();
                let pretty_cond = prettify_condition(&condition_ast);
                condition = Some((condition_ast, pretty_cond));
            } else {
                abort!(a, "Unknown attribute: {:?}", a.path.get_ident())
            }
        }

        let ty = if is_primitive_type(&f.ty, "bool") {
            if coder.is_some() {
                abort!(f, "Cannot specify coder for bool");
            }
            if default.is_none() {
                default = Some(quote! {false});
            }
            Type::Bool
        } else if is_primitive_type(&f.ty, "u32") {
            if coder.is_none() {
                abort!(f, "Must specify coder for u32");
            }
            if default.is_none() {
                default = Some(quote! {0});
            }
            let (coder, pretty_coder) = coder.unwrap();
            Type::U32(coder, pretty_coder)
        } else {
            if coder.is_some() {
                abort!(f, "Cannot specify coder for nested types");
            }
            if default.is_some() {
                abort!(f, "Cannot specify default for nested types");
            }
            let t = &f.ty;
            default = Some(quote! {#t::new()});
            Type::Nested(quote! {#t}.to_string())
        };

        let ident = f.ident.as_ref().unwrap();
        let default = default.unwrap();

        Field {
            name: ident.clone(),
            condition,
            default,
            ty,
        }
    }

    // Produce reading code (possibly with tracing).
    fn read_fun(&self, trace: bool) -> TokenStream2 {
        let ident = &self.name;
        let default = &self.default;
        let read_command = match &self.ty {
            Type::Nested(_) => quote! { self.#ident.read(br)?;},
            Type::Bool => quote! { self.#ident = Bool{}.read(br)?;},
            Type::U32(coder, _) => quote! { self.#ident = #coder.read(br)?; },
        };
        let pre_read_trace = match (&self.ty, trace) {
            (_, false) => quote! {},
            (Type::Nested(nested), true) => {
                quote! { eprintln!("reading nested {} {}", #nested, stringify!(#ident)); }
            }
            (Type::Bool, true) => quote! { eprint!("reading bool {}: ", stringify!(#ident)); },
            (Type::U32(_, pretty_coder), true) => {
                quote! { eprint!("reading {} with {}: ", stringify!(#ident), #pretty_coder); }
            }
        };
        let post_read_trace = match (&self.ty, trace) {
            (Type::Nested(_), true) | (_, false) => quote! {},
            _ => quote! { eprintln!("{}", self.#ident); },
        };
        let post_default_trace = match (&self.ty, trace) {
            (_, false) => quote! {},
            (Type::Nested(_), true) => {
                quote! { eprintln!("setting {} to default value", stringify!(#ident)); }
            }
            (_, true) => {
                quote! { eprintln!("setting {} to default value {}", stringify!(#ident), #default); }
            }
        };
        let cond_true_trace = match (&self.condition, trace) {
            (Some((_, pretty_cond)), true) => quote! { eprint!("{} is true, ", #pretty_cond); },
            _ => quote! {},
        };
        let cond_false_trace = match (&self.condition, trace) {
            (Some((_, pretty_cond)), true) => quote! { eprint!("{} is false, ", #pretty_cond); },
            _ => quote! {},
        };
        if let Some((cond, _)) = &self.condition {
            quote! {
                if #cond {
                    #cond_true_trace
                    #pre_read_trace
                    #read_command
                    #post_read_trace
                } else {
                    self.#ident = #default;
                    #cond_false_trace
                    #post_default_trace;
                }
            }
        } else {
            quote! {
                #pre_read_trace
                #read_command
                #post_read_trace
            }
        }
    }

    fn init_fun(&self) -> TokenStream2 {
        let default = &self.default;
        let ident = &self.name;
        quote! {#ident : #default}
    }

    #[cfg(feature = "tex")]
    fn texify(&self) -> String {
        let mut ret = String::new() + "    ";
        let minted = "\\mintinline[breaklines]{rust}{";
        if let Some((_, pretty_cond)) = &self.condition {
            ret += minted;
            ret += &pretty_cond.replace("self.", "");
            ret += "}";
        }
        ret += " & ";
        ret += &match &self.ty {
            Type::Nested(nested) => "\\hyperref[hdr:".to_owned() + nested + "]{" + nested + "}",
            Type::Bool => minted.to_owned() + "bool }",
            Type::U32(_, pretty_coder) => minted.to_owned() + pretty_coder + " }",
        };
        ret += " & ";
        ret += &match &self.ty {
            Type::Nested(_) => String::new(),
            _ => minted.to_owned() + &self.default.to_string() + " }",
        };
        ret += " & ";
        ret += minted;
        let ident = &self.name;
        ret += &quote! {#ident}.to_string();
        ret += "} \\\\";
        ret
    }
}

#[cfg(feature = "tex")]
fn texify(name: &str, fields: &[Field]) -> () {
    let mut table = String::new();
    table += &format!(
        "\\begin{{table}}[h]\n  \\caption{{{} bundle. \\label{{hdr:{}}}}}\n",
        name, name
    );
    table += r#"
  \centering
  % also consider >{\centering\arraybackslash}m{0.3\textwidth}
  \begin{tabular}{cccc}
    \toprule
    \bf condition & \bf type & \bf default & \bf name \\
    \midrule
"#;
    for (i, f) in fields.iter().enumerate() {
        if i != 0 {
            table += "    \\noalign{\\color{gray!50}\\hrule height 0.1pt}\n";
        }
        table += &f.texify();
    }
    table += r#"
    \bottomrule
  \end{tabular}
\end{table}"#;
    // TODO(veluca93): this may be problematic.
    fs::create_dir_all("tex").unwrap();
    let fname = format!("tex/{}.tex", name.to_owned());
    fs::write(fname, table).unwrap();
}

#[cfg(not(feature = "tex"))]
fn texify(_: &str, _: &[Field]) -> () {}

fn derive_struct(input: DeriveInput) -> TokenStream2 {
    let name = &input.ident;

    let trace = input.attrs.iter().any(|a| a.path.is_ident("trace"));

    let data = if let syn::Data::Struct(struct_data) = &input.data {
        struct_data
    } else {
        abort!(input, "derive_struct didn't get a struct");
    };

    let fields = if let syn::Fields::Named(syn::FieldsNamed {
        brace_token: _,
        named,
    }) = &data.fields
    {
        named
    } else {
        abort!(data.fields, "only named fields are supported (for now?)");
    };

    let fields: Vec<Field> = fields.iter().map(Field::parse).collect();
    let fields_read = fields.iter().map(|x: &Field| x.read_fun(trace));
    let fields_init = fields.iter().map(|x: &Field| x.init_fun());

    texify(&quote! {#name}.to_string(), &fields);

    quote! {
        impl #name {
            pub fn new() -> #name {
                #name {
                    #(#fields_init),*
                }
            }
        }
        impl crate::headers::JxlHeader for #name {
            fn read(&mut self, br: &mut BitReader) -> Result<(), Error> {
                #(#fields_read)*
                Ok(())
            }
        }
    }
}

#[proc_macro_error]
#[proc_macro_derive(JxlHeader, attributes(trace, coder, condition, default))]
pub fn derive_jxl_headers(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match &input.data {
        syn::Data::Struct(_) => derive_struct(input).into(),
        _ => abort!(input, "Only implemented for struct"),
    }
}
