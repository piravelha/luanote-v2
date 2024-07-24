from dataclasses import dataclass
from typing import TypeAlias, overload, Any


# ---------- BASIC TYPES ---------- #

Type: TypeAlias = 'TypeVar | RecVar | TypeConstructor | NamedType'
Recursive: TypeAlias = 'TypeConstructor | NamedType'

var_repr_count = 0
var_repr_map: dict[str, int] = {}
def var_repr(index: str):
  global var_repr_count, var_repr_map
  if index not in var_repr_map:
    var_repr_map[index] = var_repr_count
    var_repr_count += 1
  i = var_repr_map[index]
  letters = "abcdefghijklmnopqrstuvwxyz"
  mod = i % 26
  div = i // 26 or ""
  return letters[mod] + str(div)

class TypeVar:
  name: str
  def __init__(self, name: str = ""):
    name = name or new_var()
    self.name = name
  def __repr__(self):
    return var_repr(self.name)

class RecVar:
  name: str
  def __init__(self, name: str = ""):
    name = name or new_var()
    self.name = name
  def __repr__(self):
    return var_repr(self.name)

@dataclass
class TypeConstructor:
  name: str
  args: list[Type]
  this: RecVar
  def __repr__(self):
    if self.args:
      is_rec = any(
        contains(a, self.this.name)
        for a in self.args[:-1]) or contains(
          self.args[-1], self.this.name)
    else: is_rec = False
    rec = is_rec and f"<{self.this}>" or ""
    if self.name == "Func":
      args = ", ".join([
        repr(a) for a in self.args[:-1]
      ])
      return f"{rec}({args}) -> {self.args[-1]}"
    if self.name == "Object":
      fields = ", ".join([
        repr(a) for a in self.args
      ])
      return f"{rec}" + "{" + fields + "}"
    if self.args:
      args = ", ".join([
        repr(a) for a in self.args])
      return f"{rec}{self.name}({args})"
    return f"{self.name}"

@dataclass
class NamedType:
  name: str
  type: Type
  this: RecVar
  def __repr__(self):
    rec = contains(
      self.type,
      self.this.name
    ) and f"<{self.this}>" or ""
    return f"{rec}{self.name}: {self.type}"

var_count = 0
def new_var() -> str:
  global var_count
  var_count += 1
  return f"{var_count}"

IntType = TypeConstructor(
  "Int", [], RecVar())
BoolType = TypeConstructor(
    "Bool", [], RecVar())


# ---------- SUBSTITUTIONS ---------- #

class Substitution:
  mapping: dict[str, Type]
  def __init__(self, mapping = {}):
    new = {}
    for k, v in mapping.items():
      if isinstance(v, (str, tuple)):
        new[k] = to_type(v)
      else:
        new[k] = v
    self.mapping = new
  def apply(self, t: Type) -> Type:
    if isinstance(t, TypeVar):
      return self.mapping.get(t.name, t)
    if isinstance(t, RecVar):
      return t
    if isinstance(t, TypeConstructor):
      return TypeConstructor(
        t.name,
        [self.apply(a) for a in t.args],
        t.this)
    if isinstance(t, NamedType):
      return NamedType(
        t.name,
        self.apply(t.type),
        t.this)
  def combine(self, other: 'Substitution'):
    return Substitution({
      **self.mapping, **other.mapping})
  def __repr__(self):
    return "{" + ", ".join([
        f"{var_repr(k)} ⇒ {v}"
        for k, v in self.mapping.items()
      ]) + "}"


# ---------- REPLACE ---------- #

def replace(
    t1: Type,
    t2: Type,
    var: str = "") -> Type:
  if isinstance(t1, (TypeConstructor, NamedType)):
    var = var or t1.this.name
  if isinstance(t1, TypeVar):
    return t1
  if isinstance(t1, RecVar):
    if t1.name == var:
      return t2
    return t1
  if isinstance(t1, TypeConstructor):
    return TypeConstructor(
      t1.name,
      [replace(a, t2, var) for a in t1.args],
      t1.this)
  return NamedType(
    t1.name,
    replace(t1.type, t2, var),
    t1.this)


# ---------- UNIFY ---------- #

class UnificationException(Exception): ...

def unify(t1_, t2_) \
    -> Substitution:
  t1 = to_type(t1_)
  t2 = to_type(t2_)
  if isinstance(t1, TypeVar) \
      and isinstance(t2, TypeVar) \
      and t1.name == t2.name:
    return Substitution()
  if isinstance(t1, TypeVar):
    return Substitution({t1.name: t2})
  if isinstance(t2, TypeVar):
    return Substitution({t2.name: t1})
  if isinstance(t1, RecVar) \
      and isinstance(t2, RecVar) \
      and t1.name == t2.name:
    return Substitution()
  if isinstance(t1, TypeConstructor) \
      and isinstance(t2, TypeConstructor) \
      and t1.name == t2.name \
      and len(t1.args) == len(t2.args):
    s = Substitution()
    for a, b in zip(t1.args, t2.args):
      s = s.combine(unify(a, b))
    return s
  if isinstance(t1, NamedType) \
      and isinstance(t2, NamedType) \
      and t1.name == t2.name:
    return unify(t1.type, t2.type)
  raise UnificationException(f"'{t1}', '{t2}'")


# ---------- CONTAINS ---------- #

def contains(t: Type, var: str) -> bool:
  if isinstance(t, TypeVar):
    return False
  if isinstance(t, RecVar):
    return t.name == var
  if isinstance(t, TypeConstructor):
    return any(
      contains(a, var) for a in t.args)
  if isinstance(t, NamedType):
    return contains(t.type, var)


# ---------- EXPAND ---------- #

def expand(t: Type, *args) -> Type:
  if isinstance(t, TypeConstructor):
    i, *_ = args
    arg: Type = t.args[i]
  elif isinstance(t, NamedType):
    arg = t.type
  else:
    return t
  if isinstance(arg, TypeVar):
    return arg 
  if isinstance(arg, RecVar):
    if arg.name == t.this.name:
      return t
    return arg
  return replace(arg, t, t.this.name)
      

# ---------- UTIL ---------- #

@overload
def to_type(value: str) \
  -> Type: ...
@overload
def to_type(value: tuple[Any, list]) \
  -> TypeConstructor: ...
@overload
def to_type(value: tuple[Any, list, RecVar]) \
  -> TypeConstructor: ...
@overload
def to_type(value: 'dict[str, Any]') \
    -> NamedType: ...
@overload
def to_type(value: Type) \
  -> Type: ...
def to_type(value: Any) -> Type:
  if isinstance(value, str):
    if value.startswith("*"):
      return RecVar(value[1:])
    if value[0].isupper():
      return TypeConstructor(
        value, [], RecVar())
    return TypeVar(value)
  if isinstance(value, tuple):
    name, *rest = value
    if len(rest) == 2 \
        and isinstance(rest[-1], str) \
        and rest[-1].startswith("*"):
      rest, this = rest
      this = RecVar(this[1:])
    elif len(rest) == 2:
      rest, this = rest
    else:
      rest = rest[0]
      this = RecVar()
    rest = [to_type(r) for r in rest]
    return TypeConstructor(name, rest, this)
  if isinstance(value, dict):
    k: Any = None
    v: Any = None
    for k, v in value.items(): break
    if isinstance(v, tuple):
      if len(v) == 2:
        v, this = v
      else:
        v = v[0]
        this = RecVar()
    else:
      this = RecVar()
    return NamedType(k, to_type(v), this)
  return value


# ---------- TESTING ---------- #

Person = to_type((
  "Func", ["String", "Int", (
    "Object", [
      {"name": "String"},
      {"age": "Int"},
      {"clone": ((
        "Func", ["*Person", "*Person"],
      ),)},
    ],
    "*Person",
  )],
))

print(Person)
