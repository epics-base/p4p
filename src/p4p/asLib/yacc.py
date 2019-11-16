import warnings
from .lex import tokens, ACFError

start = 'asconfig'

def p_item(p):
    """asconfig_item : uag
                     | hag
                     | asg
       rule_item : uag_ref
                 | hag_ref
                 | calc_head
    """
    p[0] = p[1]

def p_top_items(p):
    """uag : UAG  '(' STRING ')' string_body
       hag : HAG  '(' STRING ')' string_body
       asg : ASG  '(' STRING ')' asg_body
    """
    p[0] = (p[1], p[3], p[5])

def p_list_append(p):
    """asconfig    : asconfig        asconfig_item
       string_list : string_list ',' STRING
       asg_list    : asg_list        asg_item
       rule_list   : rule_list       rule_item
    """
    #p[0] = p[1] + [p[len(p)-1]]
    p[1].append(p[len(p)-1])
    p[0] = p[1]

def p_list_one(p):
    """asconfig    : asconfig_item
       string_list : STRING
       asg_list    : asg_item
       rule_list   : rule_item
    """
    p[0] = [p[1]]

def p_head(p):
    """uag_ref  : UAG  '(' string_list ')'
       hag_ref  : HAG  '(' string_list ')'
       calc_head : CALC '(' STRING ')'
    """
    p[0] = (p[1], p[3])

def p_body_empty(p):
    """string_body :
       asg_body    :
       rule_body   :
    """
    p[0] = None

def p_body(p):
    """string_body : '{' string_list '}'
       asg_body    : '{' asg_list    '}'
       rule_body   : '{' rule_list   '}'
    """
    p[0] = p[2]

def p_asg_inp(p):
    """asg_item : INP '(' STRING ')'
    """
    p[0] = ('INP', p[1][-1:], p[3])

def p_asg_rule(p):
    """asg_item : RULE '(' INTEGER ',' STRING trap ')' rule_body
    """
    p[0] = ('RULE', p[3], p[5], p[6], p[8])

def p_trap_empty(p):
    """trap : 
    """
    p[0] = False

def p_trap(p):
    """trap : ',' STRING
    """
    if p[2] not in ('TRAPWRITE', 'NOTRAPWRITE'):
        warnings.warn("trap spec. must be 'TRAPWRITE' or 'NOTRAPWRITE'")
        p[0] = False
    else:
        p[0] = p[2]=='TRAPWRITE'

def p_error(p):
    raise ACFError("Syntax error on line %d at or before '%s'"%(p.lineno, p.value))

def parse(acf, debug=False):
    from ply import yacc, lex
    from . import lex as _lex
    lex = lex.lex(module=_lex)
    parser = yacc.yacc(debug=debug,write_tables=False)
    return parser.parse(acf, lex, debug=debug)

if __name__=='__main__':
    import sys
    from pprint import pprint
    with open(sys.argv[1],'r') as F:
        data = F.read()
    pprint(parse(data, debug=True))
