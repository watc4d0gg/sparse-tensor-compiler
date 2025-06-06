open Result
open Einsum
open Lexing

let lexer_position lexbuf =
  let pos = lexbuf.lex_curr_p in
  Printf.sprintf "%s:%d:%d" pos.pos_fname pos.pos_lnum (pos.pos_cnum - pos.pos_bol + 1)


let parse_ast filename =
  let* in_file =
    try open_in filename |> ok with
    | _ -> error (Printf.sprintf "Unable to open %s" filename)
  in
  let lexbuf = Lexing.from_channel in_file in
  lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = filename };
  let* ast =
    try Parser.prog Lexer.read lexbuf |> ok with
    | SyntaxError msg ->
      In_channel.close in_file;
      error (Printf.sprintf "%s: %s" (lexer_position lexbuf) msg)
    | Parser.Error ->
      In_channel.close in_file;
      error (Printf.sprintf "%s: syntax error" (lexer_position lexbuf))
  in
  In_channel.close in_file;
  let* _ = check_tensors Set.empty ast in
  let* _ = check_formats Map.empty ast in
  ok ast