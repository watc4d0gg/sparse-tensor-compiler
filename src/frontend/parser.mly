%parameter <Env : Core.Environment> 

%{
  open Mlir
  open Core
  open Core.Einsum(Env)

  let index_order = ref []

  let init_order value =
    index_order := [];
    value

  let compute_order index_var = match List.find_index (String.equal index_var) !index_order with
  | Some index -> index
  | None -> index_order := !index_order @ [index_var]; List.length !index_order - 1
%}

%token <float> CONST
%token <string> ID
%token <string> NAMED_FORMAT
%token <int> DIM_ID
%token TENSOR_DECL
%token LEVELS_DECL

%token PLUS
%token MINUS
%token TIMES
%token CEILDIV
%token FLOORDIV
%token MOD
%token LPAREN
%token RPAREN
%token LBRACKET
%token RBRACKET
%token LBRACE
%token RBRACE
%token ARROW

%token DENSE
%token BATCH
%token COMPRESSED
%token LOOSE_COMPRESSED
%token SINGLETON
%token STRUCTURED

%token UNORDERED
%token NON_UNIQUE
%token STRUCT_OF_ARRAYS

%token COMMA
%token COLON
%token EQUALS
%token EOF

%left PLUS MINUS
%left TIMES CEILDIV FLOORDIV
%left MOD
%nonassoc UMINUS

%start <Core.Einsum(Env).t> prog

%%

prog:
    | stmts = stmts; EOF { stmts }

stmts:
    | stmt = stmt                   { stmt }
    | first = stmts; second = stmt  { `Sequence (first, second) |> validate }

stmt:
    | tensor = ID; COLON; fmt = format                                                          { `Declare (tensor, init_order fmt) }
    | tensor = ID; LPAREN; subscript = separated_list(COMMA, index); RPAREN; EQUALS; e = expr   { if List.is_empty subscript then SyntaxError (Printf.sprintf "Missing subscript indices for the assignment of %s" tensor) |> raise else `Assign (`Access (tensor, subscript), init_order e) }
    | tensor = ID; EQUALS; expr                                                                 { SyntaxError (Printf.sprintf "Missing subscript indices for the assignment to %s" tensor) |> raise }                                                                                 

expr:
    | c = CONST                                                             { `Constant c }
    | tensor = ID; LPAREN; subscript = separated_list(COMMA, index); RPAREN { if List.is_empty subscript then SyntaxError (Printf.sprintf "Missing subscript indices for the access to %s" tensor) |> raise else `Access (tensor, subscript) }
    | tensor = ID                                                           { SyntaxError (Printf.sprintf "Missing subscript indices for the access to %s" tensor) |> raise }
    | MINUS; e = expr %prec UMINUS                                          { `Unary (TensorExpr.SUB, e) }
    | e1 = expr; PLUS; e2 = expr                                            { `Binary (TensorExpr.ADD, e1, e2) }
    | e1 = expr; MINUS; e2 = expr                                           { `Binary (TensorExpr.SUB, e1, e2) }
    | e1 = expr; TIMES; e2 = expr                                           { `Binary (TensorExpr.MUL, e1, e2) }
    | LPAREN; e = expr; RPAREN                                              { e }

dim:
    | i = ID        { AffineExpr.dim context (compute_order i) }
    | d = DIM_ID    { AffineExpr.dim context d }

index:
    | d = dim                       { d }
    | c = CONST; TIMES; d = dim     { if Float.is_integer c then AffineExpr.mul (AffineExpr.constant context (Float.to_int c)) d else SyntaxError (Printf.sprintf "Index expression requires an int, not %f" c) |> raise }
    | d = dim; TIMES; c = CONST     { if Float.is_integer c then AffineExpr.mul d (AffineExpr.constant context (Float.to_int c))  else SyntaxError (Printf.sprintf "Index expression requires an int, not %f" c) |> raise } 
    | c = CONST                     { if Float.is_integer c then AffineExpr.constant context (Float.to_int c) else SyntaxError (Printf.sprintf "Index expression requires an int, not %f" c) |> raise }
    | i1 = index; PLUS; i2 = index  { AffineExpr.add i1 i2 };

format:
    | name = NAMED_FORMAT                                                                                                                                                               { let format = String.sub name 1 (String.length name - 1) in match StringMap.find_opt format named_formats with | Some fmt -> fmt | None -> SyntaxError (Printf.sprintf "Unknown tensor format %s!" format) |> raise }
    | TENSOR_DECL; LBRACE; dims = delimited(LPAREN, separated_list(COMMA, dim), RPAREN); ARROW; LEVELS_DECL; levels = delimited(LBRACE, separated_list(COMMA, level), RBRACE); RBRACE   { let (exprs, levels) = List.split levels and order = List.length dims in if List.for_all (function | AffineExpr.Dim (i, _) -> i < order | _ -> true) dims then SparseTensorEncodingAttr.get context levels (AffineMap.get context order 0 exprs) None None None None None else SyntaxError (Printf.sprintf "Tensor dimensions have an incompatible order (above or equal to %d)!" order) |> raise }

level:
    | expr = delimited(LPAREN, coord_expr, RPAREN); COLON; level_type = level_type { (expr, level_type) }

level_type:
    | DENSE                                                                                                 { LevelType.dense () }
    | BATCH                                                                                                 { LevelType.batch () }
    | COMPRESSED; props = loption(delimited(LPAREN, separated_list(COMMA, level_property), RPAREN))         { LevelType.compressed props }
    | LOOSE_COMPRESSED; props = loption(delimited(LPAREN, separated_list(COMMA, level_property), RPAREN))   { LevelType.loose_compressed props }
    | SINGLETON; props = loption(delimited(LPAREN, separated_list(COMMA, level_property), RPAREN))          { LevelType.singleton props }
    | STRUCTURED; LBRACKET; n = CONST; COLON; m = CONST; RBRACKET                                           { if Float.is_integer n && Float.is_integer m then LevelType.structured (Float.to_int n) (Float.to_int m) else SyntaxError (Printf.sprintf "Both n and m require an int, not %f and %f" n m) |> raise }

level_property:
    | UNORDERED         { LevelNonDefaultProperty.NonOrdered }
    | NON_UNIQUE        { LevelNonDefaultProperty.NonUnique }
    | STRUCT_OF_ARRAYS  { LevelNonDefaultProperty.StructureOfArrays }

coord_expr:
    | c = CONST                                 { if Float.is_integer c then AffineExpr.constant context (Float.to_int c) else SyntaxError (Printf.sprintf "Coordinate expression requires an int, not %f" c) |> raise }
    | d = dim                                   { d }
    | l = coord_expr; PLUS; r = coord_expr      { AffineExpr.add l r }
    | l = coord_expr; TIMES; r = coord_expr     { AffineExpr.mul l r }
    | l = coord_expr; MOD; r = coord_expr       { AffineExpr.modulo l r }
    | l = coord_expr; CEILDIV; r = coord_expr   { AffineExpr.ceil_div l r }
    | l = coord_expr; FLOORDIV; r = coord_expr  { AffineExpr.floor_div l r }
    | LPAREN; e = coord_expr; RPAREN            { e }