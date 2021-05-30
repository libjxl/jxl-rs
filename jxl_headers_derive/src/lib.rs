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

#[cfg(feature = "tex")]
const THIN_LINE: &'static str = "    \\noalign{\\color{gray!50}\\hrule height 0.1pt}\n";

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
        .replace(" :: ", "::")
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

#[cfg(feature = "tex")]
fn prettify_default(d: String, ty: &str) -> String {
    d.replace(&(ty.to_owned() + " :: default()"), "")
        .replace(" :: ", "::")
}

#[cfg(feature = "tex")]
fn minted(x: &str) -> String {
    "\\mintinline[breaklines]{rust}{".to_owned() + x + "}"
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
struct U32 {
    coder: TokenStream2,
    pretty: String,
}

#[derive(Debug)]
enum Coder {
    WithoutConfig(syn::Type),
    U32(U32),
    Select(Condition, U32, U32),
}

#[derive(Debug)]
enum FieldKind {
    Unconditional(Coder),
    Conditional(Condition, Coder),
    Defaulted(Condition, Coder),
}

#[derive(Debug)]
struct Field {
    name: proc_macro2::Ident,
    kind: FieldKind,
    ty: syn::Type,
    default: Option<TokenStream2>,
}

impl Field {
    fn parse(f: &syn::Field, num: usize, all_default_field: &mut Option<syn::Ident>) -> Field {
        let mut condition = None;
        let mut default = None;
        let mut coder = None;

        let mut select_coder = None;
        let mut coder_true = None;
        let mut coder_false = None;

        let mut is_all_default = false;

        // Parse attributes.
        for a in &f.attrs {
            match a.path.get_ident().map(syn::Ident::to_string).as_deref() {
                Some("coder") => {
                    if coder.is_some() {
                        abort!(f, "Repeated coder");
                    }
                    let coder_ast = a.parse_args::<syn::Expr>().unwrap();
                    let pretty = prettify_coder(&coder_ast);
                    coder = Some(Coder::U32(U32 {
                        coder: parse_coder(coder_ast),
                        pretty,
                    }));
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
                Some("select_coder") => {
                    if select_coder.is_some() {
                        abort!(f, "Repeated select_coder");
                    }
                    let condition_ast = a.parse_args::<syn::Expr>().unwrap();
                    let pretty_cond = prettify_condition(&condition_ast);
                    select_coder = Some(Condition {
                        expr: Some(condition_ast),
                        has_all_default: false,
                        pretty: pretty_cond,
                    });
                }
                Some("coder_false") => {
                    if coder_false.is_some() {
                        abort!(f, "Repeated coder_false");
                    }
                    let coder_ast = a.parse_args::<syn::Expr>().unwrap();
                    let pretty = prettify_coder(&coder_ast);
                    coder_false = Some(U32 {
                        coder: parse_coder(coder_ast),
                        pretty,
                    });
                }
                Some("coder_true") => {
                    if coder_true.is_some() {
                        abort!(f, "Repeated coder_true");
                    }
                    let coder_ast = a.parse_args::<syn::Expr>().unwrap();
                    let pretty = prettify_coder(&coder_ast);
                    coder_true = Some(U32 {
                        coder: parse_coder(coder_ast),
                        pretty,
                    });
                }
                _ => {}
            }
        }

        if select_coder.is_some() {
            if coder_true.is_none() || coder_false.is_none() {
                abort!(
                    f,
                    "Invalid field, select_coder is set but coder_true or coder_false are not"
                )
            }
            if coder.is_some() {
                abort!(f, "Invalid field, select_coder and coder are both present")
            }
            coder = Some(Coder::Select(
                select_coder.unwrap(),
                coder_true.unwrap(),
                coder_false.unwrap(),
            ))
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

        let kind = match (condition, &default) {
            (None, _) => FieldKind::Unconditional(coder),
            (Some(cond), None) => FieldKind::Conditional(cond, coder),
            (Some(cond), Some(_)) => FieldKind::Defaulted(cond, coder),
        };
        if is_all_default {
            *all_default_field = Some(f.ident.as_ref().unwrap().clone());
        }
        Field {
            name: ident.clone(),
            kind,
            ty: f.ty.clone(),
            default,
        }
    }

    // Produces reading code (possibly with tracing).
    fn read_fun(&self, all_default_field: &Option<syn::Ident>, trace: bool) -> TokenStream2 {
        let ident = &self.name;
        let ty = &self.ty;
        let get_config = |coder: &Coder| match coder {
            Coder::WithoutConfig(_) => quote! { () },
            Coder::U32(U32 { coder, pretty: _ }) => quote! { #coder },
            Coder::Select(
                condition,
                U32 {
                    coder: coder_true,
                    pretty: _,
                },
                U32 {
                    coder: coder_false,
                    pretty: _,
                },
            ) => {
                let cnd = condition.get_expr(all_default_field).unwrap();
                quote! {
                    SelectCoder{use_true: #cnd, coder_true: #coder_true, coder_false: #coder_false}
                }
            }
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
            FieldKind::Defaulted(condition, coder) => {
                let cfg = get_config(coder);
                let cnd = condition.get_expr(all_default_field).unwrap();
                let pretty_cnd = condition.get_pretty(all_default_field);
                let default = &self.default;
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
    fn texify(&self, mut row: usize) -> String {
        let ident = &self.name;
        let ident = &quote! {#ident}.to_string();
        let (coder, condition) = match &self.kind {
            FieldKind::Unconditional(coder) => (coder, None),
            FieldKind::Conditional(condition, coder) => (coder, Some(&condition.pretty)),
            FieldKind::Defaulted(condition, coder) => (coder, Some(&condition.pretty)),
        };
        let mut ret = String::new();
        let mut add_row = |cond: Option<&str>, coder: &str, dfl: Option<&str>, ident: &str| {
            if row != 0 {
                ret += THIN_LINE;
            }
            row += 1;
            ret += &format!(
                "    {} & {} & {} & {} \\\\\n",
                cond.unwrap_or(""),
                coder,
                dfl.unwrap_or(""),
                ident
            );
        };
        let default = self.default.as_ref().map(|d| quote! { #d }.to_string());
        let default = if let Coder::WithoutConfig(ty) = &coder {
            let ty = prettify_type(ty);
            default.map(|d| prettify_default(d, &ty))
        } else {
            default
        };
        let cond = condition.as_ref().map(|x| minted(x));
        let dfl = default.as_ref().map(|x| minted(x));
        let ident = minted(ident);

        match &coder {
            Coder::WithoutConfig(ty) => {
                let ty = prettify_type(ty);
                add_row(
                    cond.as_deref(),
                    &("\\hyperref[hdr:".to_owned() + &ty + "]{" + &ty + "}"),
                    dfl.as_deref(),
                    &ident,
                );
            }
            Coder::U32(U32 { coder: _, pretty }) => {
                add_row(cond.as_deref(), &minted(pretty), dfl.as_deref(), &ident);
            }
            Coder::Select(
                Condition {
                    expr: _,
                    has_all_default: _,
                    pretty: condition,
                },
                U32 {
                    coder: _,
                    pretty: coder_true,
                },
                U32 {
                    coder: _,
                    pretty: coder_false,
                },
            ) => {
                let cond_true = if let Some(c) = &cond {
                    "(".to_owned() + condition + ") && (" + c + ")"
                } else {
                    condition.clone()
                };
                let cond_false = if let Some(c) = &cond {
                    "!(".to_owned() + condition + ") && (" + c + ")"
                } else {
                    "!(".to_owned() + condition + ")"
                };
                add_row(
                    Some(&minted(&cond_true)),
                    &minted(coder_true),
                    dfl.as_deref(),
                    &ident,
                );
                add_row(
                    Some(&minted(&cond_false)),
                    &minted(coder_false),
                    dfl.as_deref(),
                    &ident,
                );
            }
        };
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
  \begin{tabular}{>{\centering\arraybackslash}m{0.27\textwidth}>{\centering\arraybackslash}m{0.3\textwidth}>{\centering\arraybackslash}m{0.1\textwidth}>{\centering\arraybackslash}m{0.27\textwidth}}
    \toprule
    \bf condition & \bf type & \bf default & \bf name \\
    \midrule
"#;
    for (i, f) in fields.iter().enumerate() {
        table += &f.texify(i);
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
    let validate = input.attrs.iter().any(|a| a.path.is_ident("validate"));

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

    let impl_default = if fields.iter().all(|x| x.default.is_some()) {
        let defaults = fields.iter().map(|f| {
            let ident = &f.name;
            let default = f.default.as_ref().unwrap();
            quote! { #ident : #default }
        });
        quote! {
            impl #name {
                pub fn default() -> #name {
                    #name {
                        #(#defaults),*
                    }
                }
            }

        }
    } else {
        quote! {}
    };

    let impl_validate = if validate {
        quote! { return_value.check()?; }
    } else {
        quote! {}
    };

    texify(&quote! {#name}.to_string(), &fields);

    quote! {
        #impl_default
        impl crate::headers::encodings::UnconditionalCoder<()> for #name {
            fn read_unconditional(_: (), br: &mut BitReader) -> Result<#name, Error> {
                use crate::headers::encodings::UnconditionalCoder;
                use crate::headers::encodings::ConditionalCoder;
                use crate::headers::encodings::DefaultedCoder;
                #(#fields_read)*
                let return_value = #name {
                    #(#fields_names),*
                };
                #impl_validate
                Ok(return_value)
            }
        }
    }
}

#[cfg(feature = "tex")]
fn texify_enum(input: &DeriveInput) -> () {
    let name = &input.ident;
    let name = &quote! {#name}.to_string();
    let mut table = String::new();
    table += &format!(
        "\\begin{{table}}[h]\n  \\caption{{{} enum. \\label{{hdr:{}}}}}\n",
        name, name
    );
    table += r#"
  \centering
  \begin{tabular}{cc}
    \toprule
    \bf name & \bf value \\
    \midrule
"#;
    let data = if let syn::Data::Enum(enum_data) = &input.data {
        enum_data
    } else {
        abort!(input, "derive_enum didn't get a enum");
    };
    let mut last_variant = -1;
    for (row, var) in data.variants.iter().enumerate() {
        let ident = &var.ident;
        let discr = &var.discriminant;
        let n = quote! {#ident}.to_string();
        let discr = if let Some((_, d)) = discr {
            let d = quote! {#d}.to_string().parse::<i32>().unwrap();
            d
        } else {
            last_variant + 1
        };
        last_variant = discr;
        if row != 0 {
            table += THIN_LINE;
        }
        table += &format!(
            "    {} & {} \\\\\n",
            &minted(&n),
            &minted(&discr.to_string())
        );
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
fn texify_enum(_: &DeriveInput) -> () {}

fn derive_enum(input: DeriveInput) -> TokenStream2 {
    texify_enum(&input);
    let name = &input.ident;
    quote! {
        impl crate::headers::encodings::UnconditionalCoder<U32Coder> for #name {
            fn read_unconditional(config: U32Coder, br: &mut BitReader) -> Result<#name, Error> {
                use num_traits::FromPrimitive;
                let u = u32::read_unconditional(config, br)?;
                if let Some(e) =  #name::from_u32(u) {
                    Ok(e)
                } else {
                    Err(Error::InvalidEnum(u, stringify!(#name).to_string()))
                }
            }
        }
        impl crate::headers::encodings::UnconditionalCoder<()> for #name {
            fn read_unconditional(config: (), br: &mut BitReader) -> Result<#name, Error> {
                #name::read_unconditional(
                    U32Coder::Select(
                        U32::Val(0), U32::Val(1),
                        U32::BitsOffset{n: 4, off: 2},
                        U32::BitsOffset{n: 6, off: 18}), br)
            }
        }
    }
}

#[proc_macro_error]
#[proc_macro_derive(
    UnconditionalCoder,
    attributes(
        trace,
        coder,
        condition,
        default,
        all_default,
        select_coder,
        coder_true,
        coder_false,
        validate
    )
)]
pub fn derive_jxl_headers(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match &input.data {
        syn::Data::Struct(_) => derive_struct(input).into(),
        syn::Data::Enum(_) => derive_enum(input).into(),
        _ => abort!(input, "Only implemented for struct"),
    }
}
