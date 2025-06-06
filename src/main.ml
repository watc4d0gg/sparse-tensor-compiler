open! Mlir

let usage_msg = "sparseopt [-verbose] <file1> [<file2>] ... -o <output>"

let () =
  print_endline "Hello, World!";
  let input_files = ref [] in
  Arg.parse [] (fun filename -> input_files := filename :: !input_files) usage_msg;
  List.iter (fun file ->
    match Frontend.parse_ast file with
    | Ok ast -> List.iter (fun stmt -> print_endline (Einsum.fmt_stmt stmt)) ast
    | Error err -> Printf.printf "ParseError: %s\n" err) !input_files

