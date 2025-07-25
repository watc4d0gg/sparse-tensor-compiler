open Core
open Frontend
open Ir
module Einsum = Einsum (GlobalEnvironment)
module Linalg = Linalg (GlobalEnvironment)

let usage_msg = "sparse-opt [-verbose] <file1> [<file2>] ... -o <output>"

let () =
  Mlir.register_all_passes ();
  let input_files = ref [] in
  Arg.parse [] (fun filename -> input_files := filename :: !input_files) usage_msg;
  List.iter
    (fun file ->
       match parse_ast file with
       | Ok ast ->
         print_endline (Einsum.repr ast);
         let module_op = (Linalg.lower ast)#to_operation in
         print_string (Mlir.print_as_string Mlir.Ir.Operation.print module_op);
         let pass_manager = Mlir.PassManager.get Einsum.context in
         (match
            pass_manager#as_op_pass_manager#add_pipeline
              "sparse-reinterpret-map,sparsification,canonicalize,cse"
          with
          | Mlir.LogicalResult.Failure, Some error -> print_endline error
          | Mlir.LogicalResult.Failure, None -> print_endline ":///"
          | _ -> ());
         (match pass_manager#run_on_op module_op with
          | Mlir.LogicalResult.Success ->
            print_string (Mlir.print_as_string Mlir.Ir.Operation.print module_op)
          | Mlir.LogicalResult.Failure -> print_endline ":(((((")
       | Error err -> Printf.eprintf "%s\n" err)
    !input_files
