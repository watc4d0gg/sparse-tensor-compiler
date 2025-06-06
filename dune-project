(lang dune 3.19)
(using menhir 3.0)

(name sparse-tensor-compiler)

(generate_opam_files true)

(source
 (github watc4d0gg/sparse-tensor-compiler))

(authors "Matej Tomášek <tomasek.matej@gmail.com>")

(maintainers "Matej Tomášek <tomasek.matej@gmail.com>")

(license LICENSE)

; (documentation https://url/to/documentation)

(package
 (name sparse-tensor-compiler)
;  (synopsis "A short synopsis")
;  (description "A longer description")
 (depends ocaml ocamllex ocamlyacc mlirml)
 (tags
  (mlir "sparse tensor" llvm ocaml)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/reference/dune-project/index.html
