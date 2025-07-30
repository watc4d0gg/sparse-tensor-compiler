open Core
open Frontend
open Ir
module Einsum = Einsum (GlobalEnvironment)
module Linalg = Linalg (GlobalEnvironment)
module TensorRuntime = TensorRuntime (GlobalEnvironment)

let usage_msg = "sparse-opt [-verbose] <file1> [<file2>] ... -o <output>"

let () =
  Mlir.register_all_passes ();
  let input_files = ref [] in
  Arg.parse [] (fun filename -> input_files := filename :: !input_files) usage_msg;
  List.iter
    (fun file ->
       match parse_ast file with
       | Ok ast ->
         let filename = Filename.basename file |> Filename.remove_extension in
         let mlir_filename = filename ^ ".mlir" in
         let mlir_module = TensorRuntime.generate ast in
         let mlir_file = open_out mlir_filename in
         Printf.fprintf mlir_file "%s" (Mlir.print_as_string Mlir.Ir.Operation.print mlir_module#to_operation);
         close_out mlir_file;
         let _ = Sys.command ("mlir-translate --mlir-to-llvmir " ^ mlir_filename ^ " | llc -O3 --filetype=obj > " ^ filename ^ ".o") in
         ()
       | Error err -> Printf.eprintf "%s\n" err)
    !input_files
