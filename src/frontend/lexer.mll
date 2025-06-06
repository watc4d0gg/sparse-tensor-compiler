{
    open Parser
    open Einsum
    open Lexing
}

let whitespace = [' ' '\t']+
let newline = '\r' | '\n' | "\r\n"
let digit = ['0'-'9']
let frac = '.' digit*
let exp = ['e' 'E'] ['-' '+']? digit+
let constant = digit+ frac? exp?
let id = ['a'-'z' 'A'-'Z' '_'] ['a'-'z' 'A'-'Z' '0'-'9' '_']*
let format = '#' id

rule read = parse
    | whitespace    { read lexbuf }
    | newline       { new_line lexbuf; read lexbuf }
    | constant      { CONST (Lexing.lexeme lexbuf |> float_of_string) }
    | id            { ID (Lexing.lexeme lexbuf) }
    | format        { FORMAT (Lexing.lexeme lexbuf) }
    | '+'           { PLUS }
    | '-'           { MINUS }
    | '*'           { TIMES }
    | '('           { LPAREN }
    | ')'           { RPAREN }
    | ','           { COMMA }
    | ':'           { COLON }
    | '='           { EQUALS }
    | _             { raise (SyntaxError ("Unexpected char: " ^ Lexing.lexeme lexbuf)) }
    | eof           { EOF }