from .lex import tokens, ACFError

start = 'asconfig'


def _append(dst, item):
    if item is not None:
        dst.append(item)
    return dst


def p_asconfig_append(p):
    """asconfig : asconfig asconfig_item"""
    _append(p[1], p[2])
    p[0] = p[1]


def p_asconfig_one(p):
    """asconfig : asconfig_item"""
    if p[1] is None:
        p[0] = []
    else:
        p[0] = [p[1]]


def p_asconfig_item(p):
    """asconfig_item : uag_def
                     | hag_def
                     | asg_def
                     | authority_def
                     | generic_top_level_item
    """
    # generic_* and authority_def are ignored
    p[0] = p[1]


def p_uag_def(p):
    """uag_def : UAG uag_head uag_body_opt"""
    p[0] = ('UAG', p[2], p[3])


def p_hag_def(p):
    """hag_def : HAG hag_head hag_body_opt"""
    p[0] = ('HAG', p[2], p[3])


def p_asg_def(p):
    """asg_def : ASG asg_head asg_body_opt"""
    p[0] = ('ASG', p[2], p[3])


def p_uag_head(p):
    """uag_head : '(' STRING ')'"""
    p[0] = p[2]


def p_hag_head(p):
    """hag_head : '(' STRING ')'"""
    p[0] = p[2]


def p_asg_head(p):
    """asg_head : '(' STRING ')'"""
    p[0] = p[2]


def p_uag_body_opt_empty(p):
    """uag_body_opt :"""
    p[0] = None


def p_hag_body_opt_empty(p):
    """hag_body_opt :"""
    p[0] = None


def p_asg_body_opt_empty(p):
    """asg_body_opt :"""
    p[0] = None


def p_uag_body_opt(p):
    """uag_body_opt : '{' string_list '}'"""
    p[0] = p[2]


def p_hag_body_opt(p):
    """hag_body_opt : '{' string_list '}'"""
    p[0] = p[2]


def p_asg_body_opt(p):
    """asg_body_opt : '{' asg_body_list '}'"""
    p[0] = p[2]


def p_string_list_append(p):
    """string_list : string_list ',' STRING"""
    p[1].append(p[3])
    p[0] = p[1]


def p_string_list_one(p):
    """string_list : STRING"""
    p[0] = [p[1]]


def p_asg_body_list_append(p):
    """asg_body_list : asg_body_list asg_body_item"""
    _append(p[1], p[2])
    p[0] = p[1]


def p_asg_body_list_one(p):
    """asg_body_list : asg_body_item"""
    if p[1] is None:
        p[0] = []
    else:
        p[0] = [p[1]]


def p_asg_body_item(p):
    """asg_body_item : inp_config
                     | rule_config
                     | generic_asg_item
    """
    p[0] = p[1]


def p_inp_config(p):
    """inp_config : INP '(' STRING ')'"""
    # token value is e.g. 'INPA'
    p[0] = ('INP', p[1][-1:], p[3])


def p_rule_config(p):
    """rule_config : RULE rule_head rule_body_opt"""
    level, perm, trap = p[2]
    p[0] = ('RULE', level, perm, trap, p[3])


def p_rule_head(p):
    """rule_head : '(' INTEGER ',' STRING trap_opt ')'
                 | '(' INTEGER ',' STRING ')'
    """
    # NOTE: EPICS Base allows a log option string.  We currently only accept TRAPWRITE/NOTRAPWRITE.
    if len(p) == 7:
        p[0] = (p[2], p[4], p[5])
    else:
        p[0] = (p[2], p[4], False)


def p_trap_opt_empty(p):
    """trap_opt :"""
    p[0] = False


def p_trap_opt(p):
    """trap_opt : ',' STRING"""
    if p[2] not in ('TRAPWRITE', 'NOTRAPWRITE'):
        raise ACFError("Log options must be TRAPWRITE or NOTRAPWRITE")
    p[0] = (p[2] == 'TRAPWRITE')


def p_rule_body_opt_empty(p):
    """rule_body_opt :"""
    p[0] = None


def p_rule_body_opt(p):
    """rule_body_opt : '{' rule_list '}'"""
    p[0] = p[2]


def p_rule_list_append(p):
    """rule_list : rule_list rule_item"""
    _append(p[1], p[2])
    p[0] = p[1]


def p_rule_list_one(p):
    """rule_list : rule_item"""
    if p[1] is None:
        p[0] = []
    else:
        p[0] = [p[1]]


def p_rule_item(p):
    """rule_item : uag_ref
                 | hag_ref
                 | calc_ref
                 | method_ref
                 | authority_ref
                 | protocol_ref
                 | rule_generic_block_elem
    """
    p[0] = p[1]


def p_uag_ref(p):
    """uag_ref : UAG '(' string_list ')'"""
    p[0] = ('UAG', p[3])


def p_hag_ref(p):
    """hag_ref : HAG '(' string_list ')'"""
    p[0] = ('HAG', p[3])


def p_calc_ref(p):
    """calc_ref : CALC '(' STRING ')'"""
    p[0] = ('CALC', p[3])


def p_method_ref(p):
    """method_ref : METHOD '(' string_list ')'"""
    p[0] = ('METHOD', p[3])


def p_authority_ref(p):
    """authority_ref : AUTHORITY '(' string_list ')'"""
    p[0] = ('AUTHORITY', p[3])


def p_protocol_ref(p):
    """protocol_ref : PROTOCOL '(' STRING ')'"""
    p[0] = ('PROTOCOL', p[3])


# --- AUTHORITY definitions (top-level) ---


def p_authority_def(p):
    """authority_def : AUTHORITY auth_head auth_body_opt"""
    # Parsed for forward compatibility; not currently used by p4p.
    p[0] = None


def p_auth_head(p):
    """auth_head : '(' STRING ',' STRING ')'
                 | '(' STRING ')'
    """
    # Return (id, cn) or (None, cn)
    if len(p) == 6:
        p[0] = (p[2], p[4])
    else:
        p[0] = (None, p[2])


def p_auth_body_opt_empty(p):
    """auth_body_opt :"""
    p[0] = None


def p_auth_body_opt(p):
    """auth_body_opt : '{' auth_body_item_list '}'"""
    p[0] = p[2]


def p_auth_body_item_list_append(p):
    """auth_body_item_list : auth_body_item auth_body_item_list"""
    # order doesn't matter; keep list anyway
    p[2].append(p[1])
    p[0] = p[2]


def p_auth_body_item_list_one(p):
    """auth_body_item_list : auth_body_item"""
    p[0] = [p[1]]


def p_auth_body_item(p):
    """auth_body_item : AUTHORITY auth_head auth_body_opt
                      | AUTHORITY auth_head
    """
    # Parsed for forward compatibility; not currently used by p4p.
    p[0] = None


# --- Generic / future-proof syntax (parsed then ignored) ---


def p_keyword(p):
    """keyword : UAG
               | HAG
               | CALC
               | METHOD
               | AUTHORITY
               | PROTOCOL
               | non_rule_keyword
    """
    # Yield a string (used by generic constructs)
    p[0] = p[1]


def p_non_rule_keyword(p):
    """non_rule_keyword : ASG
                        | RULE
                        | INP
    """
    # INP token carries value like INPA
    p[0] = p[1]


def p_generic_top_level_item(p):
    """generic_top_level_item : STRING generic_head generic_list_block
                              | STRING generic_head generic_block
                              | STRING generic_head
    """
    # ignore silently (syntax must still be correct)
    p[0] = None


def p_generic_asg_item(p):
    """generic_asg_item : STRING generic_head generic_list_block
                        | STRING generic_head generic_block
                        | STRING generic_head
    """
    p[0] = None


def p_rule_generic_block_elem(p):
    """rule_generic_block_elem : rule_generic_block_elem_name generic_head generic_block
                               | rule_generic_block_elem_name generic_head
    """
    # Unknown predicate disables this RULE (fail-secure)
    p[0] = ('UNKNOWN', p[1])


def p_rule_generic_block_elem_name(p):
    """rule_generic_block_elem_name : non_rule_keyword
                                    | STRING
    """
    p[0] = p[1]


def p_generic_head(p):
    """generic_head : '(' ')'
                    | '(' generic_element ')'
                    | '(' generic_list ')'
    """
    p[0] = None


def p_generic_list_block(p):
    """generic_list_block : '{' generic_element '}' '{' generic_list '}'"""
    p[0] = None


def p_generic_list_append(p):
    """generic_list : generic_list ',' generic_element"""
    p[0] = None


def p_generic_list_one(p):
    """generic_list : generic_element ',' generic_element"""
    p[0] = None


def p_generic_element(p):
    """generic_element : keyword
                       | STRING
                       | INTEGER
                       | FLOAT
    """
    p[0] = None


def p_generic_block(p):
    """generic_block : '{' generic_element '}'
                     | '{' generic_list '}'
                     | '{' generic_block_list '}'
    """
    p[0] = None


def p_generic_block_list_append(p):
    """generic_block_list : generic_block_list generic_block_elem"""
    p[0] = None


def p_generic_block_list_one(p):
    """generic_block_list : generic_block_elem"""
    p[0] = None


def p_generic_block_elem(p):
    """generic_block_elem : generic_block_elem_name generic_head generic_block
                          | generic_block_elem_name generic_head
    """
    p[0] = None


def p_generic_block_elem_name(p):
    """generic_block_elem_name : keyword
                              | STRING
    """
    p[0] = None


def p_error(p):
    if p is None:
        raise ACFError("Syntax error at end of input")
    raise ACFError("Syntax error on line %d at or before '%s'" % (p.lineno, p.value))


def parse(acf, debug=False):
    from ply import yacc, lex
    from . import lex as _lex
    lexer = lex.lex(module=_lex)
    parser = yacc.yacc(debug=debug, write_tables=False)
    return parser.parse(acf, lexer, debug=debug)


if __name__ == '__main__':
    import sys
    from pprint import pprint

    with open(sys.argv[1], 'r') as F:
        data = F.read()
    pprint(parse(data, debug=True))
