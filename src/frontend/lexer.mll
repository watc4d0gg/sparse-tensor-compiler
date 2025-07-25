{
    open Core
    open Lexing
    open Tokens
}

let whitespace = [' ' '\t']+
let newline = '\r' | '\n' | "\r\n"
let digit = ['0'-'9']
let frac = '.' digit*
let exp = ['e' 'E'] ['-' '+']? digit+
let constant = digit+ frac? exp?
let id = ['a'-'z' 'A'-'Z' '_'] ['a'-'z' 'A'-'Z' '0'-'9' '_']*
let named_format = '#' id
let dim_id = 'd' digit+

rule read = parse
    | whitespace        { read lexbuf }
    | newline           { new_line lexbuf; read lexbuf }
    | constant          { CONST (Lexing.lexeme lexbuf |> float_of_string) }
    
    | dim_id            { DIM_ID (Lexing.lexeme lexbuf |> fun lex -> String.sub lex 1 (String.length lex - 1) |> int_of_string) }
    | "Tensor"          { TENSOR_DECL }
    | "Levels"          { LEVELS_DECL }
    | "ceildiv"         { CEILDIV }
    | "floordiv"        { FLOORDIV }
    | "Dense"           { DENSE }
    | "Batch"           { BATCH }
    | "Compressed"      { COMPRESSED }
    | "LooseCompressed" { LOOSE_COMPRESSED }
    | "Singleton"       { SINGLETON }
    | "Structured"      { STRUCTURED }
    | "!O"              { UNORDERED }
    | "!U"              { NON_UNIQUE }
    | "SoA"             { STRUCT_OF_ARRAYS }

    | id                { ID (Lexing.lexeme lexbuf) }
    | named_format      { NAMED_FORMAT (Lexing.lexeme lexbuf) }
    | '+'               { PLUS }
    | '-'               { MINUS }
    | '*'               { TIMES }
    | '%'               { MOD }
    | '('               { LPAREN }
    | ')'               { RPAREN }
    | '['               { LBRACKET }
    | ']'               { RBRACKET }
    | '{'               { LBRACE }
    | '}'               { RBRACE }
    (* | '<'               { LT }
    | '>'               { GT } *)
    | "->"              { ARROW }
    | ','               { COMMA }
    | ':'               { COLON }
    | '='               { EQUALS }
    | _                 { SyntaxError ("Unexpected char: " ^ Lexing.lexeme lexbuf) |> raise }
    | eof               { EOF }