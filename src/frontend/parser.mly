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

  let check_sizes sizes = if List.for_all (fun size -> Float.is_integer size && size >= 0.0) sizes then List.map Float.to_int sizes else SyntaxError "Dimension sizes should be positive integers!" |> raise
%}

%token <float> CONST
%token <string> ID
%token <string> NAMED_FORMAT
%token <int> DIM_ID
%token TENSOR_DECL
%token MATRIX_DECL
%token VECTOR_DECL


%token PLUS
%token MINUS
%token TIMES
%token CEILDIV
%token FLOORDIV
%token MOD
%token LT
%token GT
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
    | name = NAMED_FORMAT; sizes = delimited(LT, separated_list(COMMA, CONST), GT)                                                                                                                      { let format = String.sub name 1 (String.length name - 1) and dim_sizes = check_sizes sizes in match StringMap.find_opt format named_formats with | Some fmt -> if fmt#dimensions_to_levels#dims = List.length dim_sizes then RankedTensorType.get (List.init fmt#dimensions_to_levels#dims (fun i -> ShapedType.Static (List.nth dim_sizes i))) element_type (Some fmt) location else SyntaxError (Printf.sprintf "The tensor format %s was provided an incorrect number of dimension sizes!" format) |> raise | None -> SyntaxError (Printf.sprintf "Unknown tensor format %s!" format) |> raise }
    | TENSOR_DECL; sizes = delimited(LT, separated_list(COMMA, CONST), GT); LBRACE; dims = delimited(LPAREN, separated_list(COMMA, dim), RPAREN); ARROW; levels = separated_list(COMMA, level); RBRACE  { let (levels, exprs) = List.split levels and order = List.length dims and dim_sizes = check_sizes sizes in if order <> List.length dim_sizes then SyntaxError (Printf.sprintf "The tensor dimension count (%d) and the number of dimension sizes (%d) do not match!" order (List.length dim_sizes)) |> raise else if List.for_all (function | AffineExpr.Dim (i, _) -> i < order | expr -> SyntaxError (Printf.sprintf "The dimension expression %s is not supported!" (print_as_string AffineExpr.print expr)) |> raise) dims |> not then SyntaxError (Printf.sprintf "Tensor dimensions have an incompatible order (above or equal to %d)!" order) |> raise else RankedTensorType.get (List.init order (fun i -> ShapedType.Static (List.nth dim_sizes i))) element_type (if List.for_all has_dense_semantics levels then None else Some (SparseTensorEncodingAttr.get context levels (AffineMap.get context order 0 exprs) None None None None None)) location }
    | MATRIX_DECL; sizes = delimited(LT, separated_list(COMMA, CONST), GT); LBRACE; level_types = separated_list(COMMA, level_type); RBRACE                                                             { let dim_sizes = check_sizes sizes in if List.length dim_sizes <> List.length level_types || List.length dim_sizes <> 2 || List.length level_types <> 2 then SyntaxError (Printf.sprintf "The number of dimensions sizes (%d) and level types (%d) do not match (should have been 2)!" (List.length sizes) (List.length level_types)) |> raise else RankedTensorType.get (List.init 2 (fun i -> ShapedType.Static (List.nth dim_sizes i))) element_type (if List.for_all has_dense_semantics level_types then None else Some (SparseTensorEncodingAttr.get context level_types (AffineMap.get context 2 0 (List.init 2 (AffineExpr.dim context))) None None None None None)) location }
    | VECTOR_DECL; size = delimited(LT, CONST, GT); LBRACE; level_type = level_type; RBRACE                                                                                                             { let _ = check_sizes [ size ] in RankedTensorType.get ([ ShapedType.Static (Float.to_int size) ]) element_type (if has_dense_semantics level_type then None else Some (SparseTensorEncodingAttr.get context [ level_type ] (AffineMap.get context 1 0 [ AffineExpr.dim context 0 ]) None None None None None))  location }

level:
    | level_type = level_type; expr = delimited(LBRACE, coord_expr, RBRACE) { (level_type, expr) }

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