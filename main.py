from lark import Tree, Token

from parser import parser
from infer import infer, type_repr, LUA_CONTEXT

def main():
    with open("test.lua") as f:
        code = f.read()
    tree = parser.parse(code)
    infer(tree, context=LUA_CONTEXT)

if __name__ == "__main__":
    main()