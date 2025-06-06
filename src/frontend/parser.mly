%{
    open Einsum

    let is_int v =
        let c = classify_float (fst (modf v)) in
        c == FP_zero
%}

%token <float> CONST
%token <string> ID
%token <string> FORMAT
%token PLUS
%token MINUS
%token TIMES
%token LPAREN
%token RPAREN
%token COMMA
%token COLON
%token EQUALS
%token EOF

%left PLUS MINUS
%left TIMES

%start <Einsum.ast> prog

%%

prog:
    | s = stmt; stmts = prog    { s :: stmts }
    | EOF                       { [] };

stmt:
    | id = ID; COLON; fmt = FORMAT                              { Format (id, fmt) }
    | id = ID; LPAREN; s = subscript; RPAREN; EQUALS; e = expr  { Assign (id, s, e) }
    | id = ID; EQUALS; e = expr                                 { Assign (id, [], e) };

expr:
    | c = CONST                                 { Constant c }
    | id = ID; LPAREN; s = subscript; RPAREN    { Access (id, s) }
    | id = ID;                                  { Access (id, []) }
    | MINUS; e = expr                           { Unary (SUB, e) }
    | e1 = expr; PLUS; e2 = expr                { Binary (ADD, e1, e2) }
    | e1 = expr; MINUS; e2 = expr               { Binary (SUB, e1, e2) }
    | e1 = expr; TIMES; e2 = expr               { Binary (MUL, e1, e2) }
    | LPAREN; e = expr; RPAREN                  { e };

subscript:
    s = separated_list(COMMA, index)    { s };

index:
    | i = ID                        { Var i }
    | c = CONST; TIMES; i = ID      { if is_int c then Term (Float.to_int c, i) else raise (SyntaxError (Printf.sprintf "Index expression requires an int, not %f" c)) } 
    | c = CONST                     { if is_int c then Literal (Float.to_int c) else raise (SyntaxError (Printf.sprintf "Index expression requires an int, not %f" c)) }
    | i1 = index; PLUS; i2 = index  { Affine (i1, i2) };