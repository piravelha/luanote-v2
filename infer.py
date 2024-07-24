from dataclasses import dataclass
from typing import Optional

from lark import Token, Tree
from parser import parser

type Type = TypeVariable | TypeFunction | LiteralType | ConstraintType | UnionType
type LiteralType = LiteralObjectType
type ConstraintType = TypeOperator

@dataclass
class TypeVariable:
    name: str
    returning: bool = False

@dataclass
class TypeFunction:
    name: str
    args: list[Type]
    returning: bool = False

@dataclass
class TypeOperator:
    op: str
    returning: bool = False

@dataclass
class LiteralObjectType:
    fields: dict[str, Type]
    returning: bool = False
    @staticmethod
    def from_dict(key: str, d: Type, **kwargs) -> 'LiteralObjectType':
        if isinstance(d, LiteralObjectType):
            return d
        assert isinstance(d, TypeFunction)
        if isinstance(d.args[0], TypeVariable) and isinstance(d.args[1], TypeVariable):
            kwargs["context"][key] = LiteralObjectType({})
            return LiteralObjectType({})
        raise ValueError(f"Cannot convert {d} to LiteralObjectType")
    @staticmethod
    def to_dict(d: Type) -> TypeFunction:
        assert isinstance(d, LiteralObjectType)
        if not d.fields:
            return TypeFunction("dict", [new_type_variable(), new_type_variable()])
        raise ValueError(f"Cannot convert {d} to TypeFunction")

@dataclass
class UnionType:
    left: Type
    right: Type
    returning: bool = False

@dataclass
class Substitution:
    raw: dict[str, Type]
    def __iadd__(self, other: 'Substitution') -> 'Substitution':
        self = combine(self, other)
        return self

def get_loc(t: Tree | Token) -> str:
    if isinstance(t, Token):
        return f"test.lua:{t.line}:{t.column}:"
    if t.meta.empty:
        if not t.children:
            return f"test.lua:?:?:"
        return get_loc(t.children[0])
    return f"test.lua:{t.meta.line}:{t.meta.column}:"

type_var_count = 0
type_var_mappings = {}
def type_repr(t: Type | Substitution) -> str:
    global type_var_count
    if isinstance(t, TypeVariable):
        letters = "abcdefghijklmnopqrstuvwxyz"
        if not t.name in type_var_mappings:
            mod = type_var_count % 26
            type_var_count += 1
            num = type_var_count // 26 or ""
            type_var_mappings[t.name] = f"{letters[mod]}{num}"
        return type_var_mappings[t.name]
    if isinstance(t, TypeFunction):
        if t.name == "function":
            params = ", ".join([type_repr(arg) for arg in t.args[:-1]])
            return f"({params}) -> {type_repr(t.args[-1])}"
        if t.name == "dict":
            if isinstance(t.args[0], TypeFunction) and t.args[0].name == "number":
                r = type_repr(t.args[1])
                if r.find(" ") > 0:
                    return f"({r})[]"
                return f"{r}[]"
            return f"{{[{type_repr(t.args[0])}]: {type_repr(t.args[1])}}}"
        if not t.args:
            return t.name
        return f"{t.name}<{', '.join([type_repr(a) for a in t.args])}>"
    if isinstance(t, LiteralObjectType):
        return f"{{{', '.join([f'{key}: {type_repr(value)}' for key, value in t.fields.items()])}}}"
    if isinstance(t, UnionType):
        if unifies(t.left, t.right):
            return type_repr(t.right)
        if unifies(t.right, t.left):
            return type_repr(t.left)
        if t.left == NilType:
            return f"{type_repr(t.right)}?"
        if t.right == NilType:
            return f"{type_repr(t.left)}?"
        return f"({type_repr(t.left)} | {type_repr(t.right)})"
    if isinstance(t, Substitution):
        return f"S{{{', '.join([
            f'{key} |-> {type_repr(value)}'
            for key, value in t.raw.items()
        ])}}}"
    assert False, f"Not implemented: {t}"

NumberType = TypeFunction("number", [])
StringType = TypeFunction("string", [])
BooleanType = TypeFunction("boolean", [])
NilType = TypeFunction("nil", [])
def FunctionType(*args: Type) -> TypeFunction:
    return TypeFunction("function", list(args))
def DictType(key: Type, value: Type) -> TypeFunction:
    return TypeFunction("dict", [key, value])

def apply(s: Substitution, t: Type) -> Type:
    if isinstance(t, TypeVariable):
        result = s.raw.get(t.name)
        if not result:
            return t
        return apply(s, result)
    if isinstance(t, TypeFunction):
        return TypeFunction(t.name, [apply(s, arg) for arg in t.args])
    if isinstance(t, LiteralObjectType):
        return LiteralObjectType({key: apply(s, value) for key, value in t.fields.items()})
    if isinstance(t, TypeOperator):
        return t
    if isinstance(t, UnionType):
        return UnionType(apply(s, t.left), apply(s, t.right))
    if isinstance(t, IntersectionType):
        return IntersectionType(apply(s, t.left), apply(s, t.right))
    assert False, f"Not implemented: {t}"

def combine(s1: Substitution, s2: Substitution) -> Substitution:
    return Substitution({**s1.raw, **s2.raw})

def unifies(t1: Type, t2: Type) -> bool:
    try:
        unify(t1, t2, "")
        return True
    except ValueError:
        return False

def unify(t1: Type, t2: Type, loc: str) -> Substitution:
    if isinstance(t1, TypeVariable) and isinstance(t2, TypeVariable) and t1.name == t2.name:
        return Substitution({})
    if isinstance(t1, TypeVariable):
        return Substitution({t1.name: t2})
    if isinstance(t2, TypeVariable):
        return Substitution({t2.name: t1})
    if isinstance(t1, TypeOperator):
        match t1.op:
            case x if x in ["+","-","*","/","%","<",">","<=",">="]:
                unify(NumberType, t2, loc)
            case x if x in [".."]:
                unify(StringType, t2, loc)
            case x if x in ["and","or"]:
                unify(BooleanType, t2, loc)
        return Substitution({})
    if isinstance(t2, TypeOperator):
        match t2.op:
            case x if x in ["+","-","*","/","%","<",">","<=",">="]:
                unify(NumberType, t1, loc)
            case x if x in [".."]:
                unify(StringType, t1, loc)
            case x if x in ["and","or"]:
                unify(BooleanType, t1, loc)
        return Substitution({})
    if isinstance(t1, UnionType):
        if unifies(t1.left, t2) and unifies(t1.right, t2):
            return Substitution({})
        raise ValueError(f"{loc} Expected '{type_repr(t2)}', got '{type_repr(t1)}'")
    if isinstance(t2, UnionType):
        if unifies(t1, t2.left) or unifies(t1, t2.right):
            return Substitution({})
        raise ValueError(f"{loc} Expected '{type_repr(t2)}', got '{type_repr(t1)}'")
    if isinstance(t1, TypeFunction) and isinstance(t2, TypeFunction):
        if t1.name != t2.name:
            raise ValueError(f"{loc} Expected '{t2.name}', got '{t1.name}'")
        if len(t1.args) != len(t2.args):
            raise ValueError(f"{loc} Types contain different number of generic arguments: {len(t1.args)}, and {len(t2.args)}")
        s = Substitution({})
        for arg1, arg2 in zip(t1.args, t2.args):
            s += unify(apply(s, arg1), apply(s, arg2), loc)
        return s
    if isinstance(t1, LiteralObjectType) and isinstance(t2, LiteralObjectType):
        if t1.fields.keys() != t2.fields.keys():
            raise ValueError(f"{loc} Fields do not match: {t1.fields.keys()}, and {t2.fields.keys()}")
        s = Substitution({})
        for key in t1.fields:
            s += unify(apply(s, t1.fields[key]), apply(s, t2.fields[key]), loc)
        return s
    raise ValueError(f"{loc} Expected '{type_repr(t2)}', got '{type_repr(t1)}'")

new_type_variable_counter = 0
def new_type_variable() -> TypeVariable:
    global new_type_variable_counter
    value = new_type_variable_counter
    new_type_variable_counter += 1
    return TypeVariable(f"{value}")

type Inferred = tuple[Substitution, Type]

def infer_reveal_type(expr, **kwargs) -> Inferred:
    s, t = infer(expr, **kwargs)
    print(f"{kwargs["loc"]} @reveal: {type_repr(t)}")
    return Substitution({}), NilType

def infer_number(token: Token, **kwargs) -> Inferred:
    return Substitution({}), NumberType

def infer_string(token: Token, **kwargs) -> Inferred:
    return Substitution({}), StringType

def infer_boolean(token: Token, **kwargs) -> Inferred:
    return Substitution({}), BooleanType

def infer_nil(token: Token, **kwargs) -> Inferred:
    return Substitution({}), NilType

def infer_name(token: Token, **kwargs) -> Inferred:
    return Substitution({}), kwargs["context"][token.value]

def infer_dict(*fields, **kwargs) -> Inferred:
    key_type = new_type_variable()
    value_type = new_type_variable()
    s = Substitution({})
    for field in fields:
        if len(field.children) == 1:
            expr_s, expr = infer(field.children[0], **kwargs)
            s += unify(NumberType, key_type, kwargs["loc"])
            s += unify(expr, value_type, kwargs["loc"])
            s += expr_s
            key_type = apply(s, key_type)
            value_type = apply(s, value_type)
            continue
        key, value = field.children
        key_expr_s, key_expr = infer(key, **kwargs)
        value_expr_s, value_expr = infer(value, **kwargs)
        s += unify(key_expr, key_type, kwargs["loc"])
        s += unify(value_expr, value_type, kwargs["loc"])
        s += key_expr_s
        s += value_expr_s
        key_type = apply(s, key_type)
        value_type = apply(s, value_type)
    return s, TypeFunction("dict", [key_type, value_type])

def infer_obj(*fields, **kwargs) -> Inferred:
    field_types = {}
    s = Substitution({})
    for field in fields:
        key, value = field.children
        value_expr_s, value_expr = infer(value, **kwargs)
        s += value_expr_s
        field_types[key.value] = value_expr
    return s, LiteralObjectType(field_types)

def infer_prop_expr(obj, prop, **kwargs) -> Inferred:
    obj_s, obj_expr = infer(obj, **kwargs)
    if isinstance(obj_expr, TypeVariable):
        prop_type = new_type_variable()
        s = Substitution({obj_expr.name: LiteralObjectType({prop.value: prop_type})})
        return s, prop_type
    if not isinstance(obj_expr, LiteralObjectType):
        raise ValueError(f"{kwargs["loc"]} Attempting to access property of non-object value: {type_repr(obj_expr)}")
    prop_type = obj_expr.fields.get(prop.value)
    if not prop_type:
        raise ValueError(f"{kwargs["loc"]} Object does not have property '{prop.value}'")
    return obj_s, prop_type

def infer_index_expr(obj, index, **kwargs) -> Inferred:
    obj_s, obj_expr = infer(obj, **kwargs)
    if isinstance(obj_expr, TypeVariable):
        s, index_type = infer(index, **kwargs)
        value_type = new_type_variable()
        s += Substitution({obj_expr.name: DictType(index_type, value_type)})
        s += obj_s
        return s, value_type
    if not isinstance(obj_expr, TypeFunction) or obj_expr.name != "dict":
        raise ValueError(f"{kwargs["loc"]} Attempting to index non-dictionary value: {type_repr(obj_expr)}")
    s, index_type = infer(index, **kwargs)
    value_type = obj_expr.args[1]
    s += unify(index_type, obj_expr.args[0], kwargs["loc"])
    s += obj_s
    return s, value_type

def infer_unary_expr(op, expr, **kwargs) -> Inferred:
    if op.value == "#":
        expr_s, expr_expr = infer(expr, **kwargs)
        any_dict = DictType(new_type_variable(), new_type_variable())
        s = unify(any_dict, expr_expr, kwargs["loc"])
        s += expr_s
        return s, NumberType
    if op.value == "-":
        expr_s, expr_expr = infer(expr, **kwargs)
        s = unify(NumberType, expr_expr, kwargs["loc"])
        s += expr_s
        return s, NumberType
    if op.value == "not":
        expr_s, expr_expr = infer(expr, **kwargs)
        s = unify(BooleanType, expr_expr, kwargs["loc"])
        s += expr_s
        return s, BooleanType
    assert False, f"Not implemented: {op.value}"

def infer_pow_expr(left, op, right, **kwargs) -> Inferred:
    left_s, left_expr = infer(left, **kwargs)
    right_s, right_expr = infer(right, **kwargs)
    s = unify(TypeOperator(op.value), left_expr, kwargs["loc"])
    s += unify(TypeOperator(op.value), right_expr, kwargs["loc"])
    s += unify(left_expr, right_expr, kwargs["loc"])
    s += left_s
    s += right_s
    return s, NumberType

def infer_mul_expr(left, op, right, **kwargs) -> Inferred:
    left_s, left_expr = infer(left, **kwargs)
    right_s, right_expr = infer(right, **kwargs)
    s = unify(TypeOperator(op.value), left_expr, kwargs["loc"])
    s += unify(TypeOperator(op.value), right_expr, kwargs["loc"])
    s += unify(left_expr, right_expr, kwargs["loc"])
    s += left_s
    s += right_s
    return s, NumberType

def infer_add_expr(left, op, right, **kwargs) -> Inferred:
    left_s, left_expr = infer(left, **kwargs)
    right_s, right_expr = infer(right, **kwargs)
    s = unify(TypeOperator(op.value), left_expr, kwargs["loc"])
    s += unify(TypeOperator(op.value), right_expr, kwargs["loc"])
    s += unify(left_expr, right_expr, kwargs["loc"])
    s += left_s
    s += right_s
    return s, NumberType

def infer_rel_expr(left, op, right, **kwargs) -> Inferred:
    left_s, left_expr = infer(left, **kwargs)
    right_s, right_expr = infer(right, **kwargs)
    s = unify(TypeOperator(op.value), left_expr, kwargs["loc"])
    s += unify(TypeOperator(op.value), right_expr, kwargs["loc"])
    s += unify(left_expr, right_expr, kwargs["loc"])
    s += left_s
    s += right_s
    return s, BooleanType

def infer_eq_expr(left, op, right, **kwargs) -> Inferred:
    left_s, left_expr = infer(left, **kwargs)
    right_s, right_expr = infer(right, **kwargs)
    s = unify(TypeOperator(op.value), left_expr, kwargs["loc"])
    s += unify(TypeOperator(op.value), right_expr, kwargs["loc"])
    if not unifies(left_expr, right_expr) and not unifies(right_expr, left_expr):
        raise ValueError(f"{kwargs["loc"]} Cannot compare '{type_repr(left_expr)}' with '{type_repr(right_expr)}'")
    if unifies(left_expr, right_expr):
        s += unify(left_expr, right_expr, kwargs["loc"])
    if unifies(right_expr, left_expr):
        s += unify(right_expr, left_expr, kwargs["loc"])
    s += left_s
    s += right_s
    return s, BooleanType

def infer_log_expr(left, op, right, **kwargs) -> Inferred:
    left_s, left_expr = infer(left, **kwargs)
    right_s, right_expr = infer(right, **kwargs)
    s = unify(TypeOperator(op.value), left_expr, kwargs["loc"])
    s += unify(TypeOperator(op.value), right_expr, kwargs["loc"])
    s += unify(left_expr, right_expr, kwargs["loc"])
    s += left_s
    s += right_s
    return s, BooleanType

def infer_var_decl(name, expr, **kwargs) -> Inferred:
    s, t = infer(expr, **kwargs)
    kwargs["context"][name.value] = t
    return s, t

def set_prefix_expr(tree: Tree, expr: Type, **kwargs) -> None:
    def get_type(tree: Tree, **kwargs) -> Type:
        if isinstance(tree, Token):
            return infer(tree, **kwargs)[1]
        if tree.data == "prop_expr":
            return get_type(tree.children[0], **kwargs)
        if tree.data == "index_expr":
            return get_type(tree.children[0], **kwargs)
        assert False, f"Not implemented: {tree.data}"

    path = get_type(tree, **kwargs)

    def run(tree: Tree, path: Type, **kwargs) -> tuple[Type, Type | str]:
        if isinstance(tree, Token):
            assert tree.type == "NAME"
            return path, tree.value
        old_path = path
        path, parent_path = run(tree.children[0], path, **kwargs)
        parent_path = parent_path or old_path
        if tree.data == "prop_expr":
            prop = tree.children[1]
            assert isinstance(prop, Token)
            path = LiteralObjectType.from_dict(prop.value, path, **kwargs)
            if isinstance(parent_path, str):
                kwargs["context"][parent_path] = path
            else:
                assert isinstance(parent_path, LiteralObjectType)
                parent_path.fields[prop.value] = path
            assert isinstance(path, LiteralObjectType)
            return path.fields.get(prop.value, NilType), path
        if tree.data == "index_expr":
            assert isinstance(path, TypeFunction) and path.name == "dict"
            return path.args[1], path
        assert False

    path, parent_path = run(tree, path, **kwargs)

    if tree.data == "prop_expr":
        prop = tree.children[1]
        assert isinstance(parent_path, LiteralObjectType)
        assert isinstance(prop, Token)
        parent_path.fields[prop.value] = expr
    if tree.data == "index_expr":
        s, index = infer(tree.children[1], **kwargs)
        assert isinstance(parent_path, TypeFunction) and parent_path.name == "dict"
        s += unify(parent_path.args[0], index, kwargs["loc"])
        s += unify(parent_path.args[1], expr, kwargs["loc"])
        parent_path.args[0] = apply(s, parent_path.args[0])
        parent_path.args[1] = apply(s, parent_path.args[1])

def infer_assign_stmt(prefix, expr, **kwargs) -> Inferred:
    expr_s, expr_expr = infer(expr, **kwargs)
    set_prefix_expr(prefix, expr_expr, **kwargs)
    return expr_s, expr_expr

def infer_func_body(params, body, **kwargs) -> Inferred:
    param_types = [new_type_variable() for _ in params.children]
    kwargs["context"] = kwargs["context"].copy()
    kwargs["context"].update({
        param.value: t
        for param, t in zip(params.children, param_types)
    })
    body_s, body_expr = infer(body, **kwargs)
    func_type = FunctionType(*param_types, body_expr)
    return body_s, apply(body_s, func_type)

def infer_func_expr(func, **kwargs) -> Inferred:
    return infer(func, **kwargs)

def infer_func_call(func, args, **kwargs) -> Inferred:
    arg_locs = [get_loc(arg) for arg in args.children]
    func_s, func_expr = infer(func, **kwargs)
    if isinstance(func_expr, TypeVariable):
        return_type = new_type_variable()
        param_types: list[Type] = [
            new_type_variable() for _ in args.children
        ]
        func_type = FunctionType(*param_types, return_type)
        s = Substitution({func_expr.name: func_type})
        for arg_type, param_type in zip(args.children, param_types):
            arg_s, arg_expr = infer(arg_type, **kwargs)
            s += arg_s
            s += unify(arg_expr, param_type, kwargs["loc"])
        return s, apply(s, return_type)
    if not isinstance(func_expr, TypeFunction) or func_expr.name != "function":
        raise ValueError(f"{kwargs["loc"]} Attempting to call non-function value: {type_repr(func_expr)}")
    arg_s = Substitution({})
    arg_types = []
    for arg in args.children:
        arg_expr_s, arg_expr = infer(arg, **kwargs)
        arg_s += arg_expr_s
        arg_types.append(arg_expr)
    s = Substitution({})
    new_args = func_expr.args[:-1]
    for i, (arg_type, param_type) in enumerate(zip(arg_types, new_args)):
        s += unify(arg_type, param_type, arg_locs[i])
    s += func_s
    return s, apply(s, func_expr.args[-1])    

def infer_func_decl(name, func, **kwargs) -> Inferred:
    kwargs["context"][name.value] = new_type_variable()
    func_s, func_expr = infer(func, **kwargs)
    kwargs["context"][name.value] = func_expr
    return func_s, func_expr

def infer_return_stmt(expr, **kwargs) -> Inferred:
    s, t = infer(expr, **kwargs)
    t.returning = True
    return s, t

def infer_range_for_stmt(var, start, stop, step, body, **kwargs) -> Inferred:
    start_s, start_expr = infer(start, **kwargs)
    stop_s, stop_expr = infer(stop, **kwargs)
    step_s, step_expr = infer(step, **kwargs)
    s = unify(start_expr, NumberType, get_loc(start))
    s += unify(stop_expr, NumberType, get_loc(stop))
    s += unify(step_expr, NumberType, get_loc(step))
    s += start_s
    s += stop_s
    s += step_s
    kwargs["context"] = kwargs["context"].copy()
    kwargs["context"][var.value] = NumberType
    body_s, body_expr = infer(body, **kwargs)
    s += body_s
    return s, body_expr

def gradual_if_inference_type_fn(args, right, **kwargs) -> None:
    if len(args.children) != 1: return
    arg = args.children[0]
    if not isinstance(arg, Token) or arg.type != "NAME": return
    if not isinstance(right, Token) or right.type != "STRING": return
    if right.value[1:-1] in ["number", "string", "boolean", "nil"]:
        kwargs["context"][arg.value] = TypeFunction(right.value[1:-1], [])
        return
    if right.value[1:-1] == "table":
        kwargs["context"][arg.value] = DictType(new_type_variable(), new_type_variable())
        return
    return

def gradual_if_inference(cond, **kwargs) -> None:
    if not isinstance(cond, Tree): return
    if cond.data != "eq_expr": return
    left, op, right = cond.children
    assert isinstance(op, Token)
    if isinstance(left, Tree) and left.data == "func_call" and op.value == "==":
        func, args = left.children
        if isinstance(func, Token) and func.value == "type":
            gradual_if_inference_type_fn(args, right, **kwargs)
        return
    if isinstance(left, Token) and left.type == "NAME":
        _, right_expr = infer(right, **kwargs)
        kwargs["context"][left.value] = right_expr
    return

def infer_if_stmt(cond, body, elseifs, else_branch, **kwargs) -> Inferred:
    cond_s, cond_expr = infer(cond, **kwargs)
    kwargs["context"] = kwargs["context"].copy()
    gradual_if_inference(cond, **kwargs)
    s, body_expr = infer(body, **kwargs)
    s += unify(cond_expr, BooleanType, get_loc(cond))
    s += cond_s
    returns = body_expr.returning
    for elseif in elseifs.children:
        elseif_cond, elseif_body = elseif.children
        gradual_if_inference(elseif_cond, **kwargs)
        elseif_cond_s, elseif_cond_expr = infer(elseif_cond, **kwargs)
        s += elseif_cond_s
        s += unify(elseif_cond_expr, BooleanType, get_loc(elseif_cond))
        elseif_body_s, elseif_body_expr = infer(elseif_body, **kwargs)
        if elseif_body_expr.returning:
            returns = True
        s += elseif_body_s
        s += unify(elseif_body_expr, body_expr, get_loc(elseif_body))
    if else_branch.children:
        else_body = else_branch.children[0]
        else_s, else_expr = infer(else_body, **kwargs)
        s += else_s
        s += unify(body_expr, else_expr, get_loc(else_branch))
        if else_expr.returning:
            returns = True
    body_expr.returning = returns
    return s, body_expr

def infer_chunk(*stmts, **kwargs) -> Inferred:
    stmt_locs = [get_loc(stmt) for stmt in stmts]
    s = Substitution({})
    ret = None
    for i, stmt in enumerate(stmts):
        stmt_s, stmt_expr = infer(stmt, **kwargs)
        s += stmt_s
        if stmt_expr.returning:
            if not ret:
                ret = stmt_expr
            else:
                ret_s = unify(ret, stmt_expr, stmt_locs[i])
                s += ret_s
                ret = apply(ret_s, ret)

    return s, ret or NilType

DATA_TO_INFER = {
    "reveal_type": infer_reveal_type,
    "NUMBER": infer_number,
    "STRING": infer_string,
    "BOOLEAN": infer_boolean,
    "NIL": infer_nil,
    "NAME": infer_name,
    "dict": infer_dict,
    "obj": infer_obj,
    "prop_expr": infer_prop_expr,
    "index_expr": infer_index_expr,
    "unary_expr": infer_unary_expr,
    "pow_expr": infer_pow_expr,
    "mul_expr": infer_mul_expr,
    "add_expr": infer_add_expr,
    "rel_expr": infer_rel_expr,
    "eq_expr": infer_eq_expr,
    "log_expr": infer_log_expr,
    "assign_stmt": infer_assign_stmt,
    "var_decl": infer_var_decl,
    "func_body": infer_func_body,
    "func_expr": infer_func_expr,
    "func_call": infer_func_call,
    "func_decl": infer_func_decl,
    "return_stmt": infer_return_stmt,
    "range_for_stmt": infer_range_for_stmt,
    "if_stmt": infer_if_stmt,
    "chunk": infer_chunk,
}

def infer(tree: Tree | Token, **kwargs) -> Inferred:
    kwargs["loc"] = get_loc(tree)
    if isinstance(tree, Token):
        return DATA_TO_INFER[tree.type](tree, **kwargs)
    if tree.data in DATA_TO_INFER:
        return DATA_TO_INFER[tree.data](*tree.children, **kwargs)
    assert False, f"Not implemented: {tree.data}"

LUA_CONTEXT = {
    "print": FunctionType(
        new_type_variable(),
        NilType,
    ),
    "tostring": FunctionType(
        new_type_variable(),
        StringType,
    ),
    "tonumber": FunctionType(
        StringType,
        UnionType(NumberType, NilType),
    ),
    "type": FunctionType(
        new_type_variable(),
        StringType,
    ),
}