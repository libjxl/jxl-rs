{
  inputs = {
    crane.url = "github:ipetkov/crane";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      crane,
      fenix,
      flake-utils,
      nixpkgs,
      ...
    }:

    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        fenix' = fenix.packages.${system};

        rustBuildToolchain = fenix'.stable.withComponents [
          "cargo"
          "clippy"
          "rustc"
          "rustfmt"
          "rust-src"
        ];

        rustDevToolchain = fenix'.combine [
          rustBuildToolchain
          fenix'.rust-analyzer # nightly rust-analyzer
        ];

        rustMiriToolchain = fenix'.complete.withComponents [
          "cargo"
          "clippy"
          "miri"
          "rustc"
          "rustfmt"
          "rust-analyzer"
          "rust-src"
        ];

        mkCraneLib = (crane.mkLib pkgs).overrideToolchain;

        mkShellForToolchain =
          toolchain:
          (mkCraneLib toolchain).devShell {
            checks = self.checks.${system};

            packages = with pkgs; [
              pkg-config
              nixfmt
              nixfmt-tree
              rustPlatform.bindgenHook
            ];
          };

        buildCraneLib = mkCraneLib rustBuildToolchain;

        jxlFilter = path: type: builtins.match ".*jxl$" path != null;
        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = path: type: (jxlFilter path type) || (buildCraneLib.filterCargoSources path type);
          name = "source";
        };

        commonBuildArgs = {
          pname = "jxl-rs-workspace";
          version = "0.0.0";

          inherit src;
          strictDeps = true;

          nativeBuildInputs = with pkgs; [
            pkg-config
            rustPlatform.bindgenHook
          ];
        };
        cargoArtifacts = buildCraneLib.buildDepsOnly commonBuildArgs;
      in
      {
        checks = {
          jxl-rs-clippy-all-features = buildCraneLib.cargoClippy (
            commonBuildArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets --all-features --tests --all -- -D warnings";
            }
          );

          jxl-rs-clippy-no-features = buildCraneLib.cargoClippy (
            commonBuildArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets --no-default-features --tests --all -- -D warnings";
            }
          );

          jxl-rs-nextest = buildCraneLib.cargoNextest (
            commonBuildArgs
            // {
              inherit cargoArtifacts;
            }
          );

          jxl-rs-fmt = buildCraneLib.cargoFmt {
            pname = "jxl-rs-workspace";
            version = "0.0.0";

            inherit src;
          };
        };

        # Setup fenix binary cache (https://app.cachix.org/cache/fenix) for faster builds
        devShells = {
          default = mkShellForToolchain rustDevToolchain;
          miri = mkShellForToolchain rustMiriToolchain;
        };
      }
    );
}
