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
                let coder = parse_single_coder(&input, None);
                return quote! {U32Coder::Direct(#coder)};
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
            return quote! {U32Coder::Select(#(#args),*)};
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

fn prettify_condition(cond: &syn::Expr) -> String {
    (quote! {#cond})
        .to_string()
        .replace(" . ", ".")
        .replace("! ", "!")
}

fn prettify_coder(coder: &syn::Expr) -> String {
    (quote! {#coder}).to_string()
}

#[cfg(feature = "tex")]
fn prettify_type(ty: &syn::Type) -> String {
    let mut ret = (quote! {#ty}).to_string().replace(' ', "");
    if ret.starts_with("Option<") {
        ret = ret[7..ret.len() - 1].to_owned();
    }
    return ret;
}

#[derive(Debug)]
enum Coder {
    WithoutConfig(syn::Type),
    U32(TokenStream2, String),
}

#[derive(Debug)]
struct Condition {
    expr: Option<syn::Expr>,
    has_all_default: bool,
    pretty: String,
}

impl Condition {
    fn get_expr(&self, all_default_field: &Option<syn::Ident>) -> Option<TokenStream2> {
        if self.has_all_default {
            let all_default = all_default_field.as_ref().unwrap();
            match &self.expr {
                Some(expr) => Some(quote! { !#all_default && (#expr) }),
                None => Some(quote! { !#all_default }),
            }
        } else {
            match &self.expr {
                Some(expr) => Some(quote! { #expr }),
                None => None,
            }
        }
    }
    fn get_pretty(&self, all_default_field: &Option<syn::Ident>) -> String {
        if self.has_all_default {
            let all_default = all_default_field.as_ref().unwrap();
            let all_default = "!".to_owned() + &quote! {#all_default}.to_string();
            match &self.expr {
                Some(_) => all_default + " && (" + &self.pretty + ")",
                None => all_default,
            }
        } else {
            self.pretty.clone()
        }
    }
}

#[derive(Debug)]
enum FieldKind {
    Unconditional(Coder),
    Conditional(Condition, Coder),
    Defaulted(TokenStream2, Condition, Coder),
}

#[derive(Debug)]
struct Field {
    name: proc_macro2::Ident,
    kind: FieldKind,
    ty: syn::Type,
}

impl Field {
    fn parse(f: &syn::Field, num: usize, all_default_field: &mut Option<syn::Ident>) -> Field {
        let mut condition = None;
        let mut default = None;
        let mut coder = None;

        let mut is_all_default = false;

        // Parse attributes.
        for a in &f.attrs {
            match a.path.get_ident().map(syn::Ident::to_string).as_deref() {
                Some("coder") => {
                    if coder.is_some() {
                        abort!(f, "Repeated coder");
                    }
                    let coder_ast = a.parse_args::<syn::Expr>().unwrap();
                    let pretty_coder = prettify_coder(&coder_ast);
                    coder = Some(Coder::U32(parse_coder(coder_ast), pretty_coder));
                }
                Some("default") => {
                    if default.is_some() {
                        abort!(f, "Repeated default");
                    }
                    let default_expr = a.parse_args::<syn::Expr>().unwrap();
                    default = Some(quote! {#default_expr});
                }
                Some("condition") => {
                    if condition.is_some() {
                        abort!(f, "Repeated condition");
                    }
                    let condition_ast = a.parse_args::<syn::Expr>().unwrap();
                    let pretty_cond = prettify_condition(&condition_ast);
                    condition = Some(Condition {
                        expr: Some(condition_ast),
                        has_all_default: all_default_field.is_some(),
                        pretty: pretty_cond,
                    });
                }
                Some("all_default") => {
                    if num != 0 {
                        abort!(f, "all_default is not the first field");
                    }
                    is_all_default = true;
                }
                _ => {}
            }
        }

        let condition = if condition.is_some() || all_default_field.is_none() {
            condition
        } else {
            Some(Condition {
                expr: None,
                has_all_default: true,
                pretty: String::new(),
            })
        };

        // Assume nested field if no coder.
        let coder = coder.unwrap_or(Coder::WithoutConfig(f.ty.clone()));

        let ident = f.ident.as_ref().unwrap();

        let kind = match (condition, default) {
            (None, None) => FieldKind::Unconditional(coder),
            (Some(cond), None) => FieldKind::Conditional(cond, coder),
            (None, Some(_)) => {
                abort!(f, "Field has default but no condition");
            }
            (Some(cond), Some(def)) => FieldKind::Defaulted(def, cond, coder),
        };
        if is_all_default {
            *all_default_field = Some(f.ident.as_ref().unwrap().clone());
        }
        Field {
            name: ident.clone(),
            kind,
            ty: f.ty.clone(),
        }
    }

    // Produces reading code (possibly with tracing).
    fn read_fun(&self, all_default_field: &Option<syn::Ident>, trace: bool) -> TokenStream2 {
        let ident = &self.name;
        let ty = &self.ty;
        let get_config = |coder: &Coder| match coder {
            Coder::WithoutConfig(_) => quote! { () },
            Coder::U32(coder, _) => quote! { #coder },
        };
        match &self.kind {
            FieldKind::Unconditional(coder) => {
                let cfg = get_config(coder);
                let trc = if trace {
                    quote! { eprintln!("Setting {} to {:?}", stringify!(#ident), #ident); }
                } else {
                    quote! {}
                };
                quote! {
                    let #ident = <#ty>::read_unconditional(#cfg, br)?;
                    #trc
                }
            }
            FieldKind::Conditional(condition, coder) => {
                let cfg = get_config(coder);
                let cnd = condition.get_expr(all_default_field).unwrap();
                let pretty_cnd = condition.get_pretty(all_default_field);
                let trc = if trace {
                    quote! { eprintln!("{} is {}, setting {} to {:?}", #pretty_cnd, #cnd, stringify!(#ident), #ident); }
                } else {
                    quote! {}
                };
                quote! {
                    let #ident = <#ty>::read_conditional(#cfg, #cnd, br)?;
                    #trc
                }
            }
            FieldKind::Defaulted(default, condition, coder) => {
                let cfg = get_config(coder);
                let cnd = condition.get_expr(all_default_field).unwrap();
                let pretty_cnd = condition.get_pretty(all_default_field);
                let trc = if trace {
                    quote! { eprintln!("{} is {}, setting {} to {:?}", #pretty_cnd, #cnd, stringify!(#ident), #ident); }
                } else {
                    quote! {}
                };
                quote! {
                    let #ident = <#ty>::read_defaulted(#cfg, #cnd, #default, br)?;
                    #trc
                }
            }
        }
    }

    #[cfg(feature = "tex")]
    fn texify(&self) -> String {
        let mut ret = String::new() + "    ";
        let minted = "\\mintinline[breaklines]{rust}{";
        let (coder, condition, default) = match &self.kind {
            FieldKind::Unconditional(coder) => (coder, None, None),
            FieldKind::Conditional(condition, coder) => (coder, Some(&condition.pretty), None),
            FieldKind::Defaulted(default, condition, coder) => {
                (coder, Some(&condition.pretty), Some(default.to_string()))
            }
        };
        if let Some(pretty_cond) = condition {
            ret += minted;
            ret += &pretty_cond;
            ret += "}";
        }
        ret += " & ";
        ret += &match &coder {
            Coder::WithoutConfig(ty) => {
                let ty = prettify_type(ty);
                "\\hyperref[hdr:".to_owned() + &ty + "]{" + &ty + "}"
            }
            Coder::U32(_, pretty_coder) => minted.to_owned() + pretty_coder + " }",
        };
        ret += " & ";
        if let Some(dfl) = default {
            ret += &(minted.to_owned() + &dfl + " }")
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
  \begin{tabular}{>{\centering\arraybackslash}m{0.25\textwidth}>{\centering\arraybackslash}m{0.4\textwidth}>{\centering\arraybackslash}m{0.1\textwidth}>{\centering\arraybackslash}m{0.2\textwidth}}
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

    let mut all_default_field = None;

    let fields: Vec<_> = fields
        .iter()
        .enumerate()
        .map(|(n, f)| Field::parse(f, n, &mut all_default_field))
        .collect();
    let fields_read = fields.iter().map(|x| x.read_fun(&all_default_field, trace));
    let fields_names = fields.iter().map(|x| &x.name);

    texify(&quote! {#name}.to_string(), &fields);

    quote! {
        impl crate::headers::JxlHeader for #name {
            fn read(br: &mut BitReader) -> Result<#name, Error> {
                use crate::headers::encodings::UnconditionalCoder;
                use crate::headers::encodings::ConditionalCoder;
                use crate::headers::encodings::DefaultedCoder;
                #(#fields_read)*
                Ok(#name {
                    #(#fields_names),*
                })
            }
        }
    }
}

#[proc_macro_error]
#[proc_macro_derive(JxlHeader, attributes(trace, coder, condition, default, all_default))]
pub fn derive_jxl_headers(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match &input.data {
        syn::Data::Struct(_) => derive_struct(input).into(),
        _ => abort!(input, "Only implemented for struct"),
    }
}
