from ply import lex

class ACFError(RuntimeError):
    pass

tokens = (
    'UAG',
    'HAG',
    'ASG',
    'RULE',
    'CALC',
    'INP',
    'INTEGER',
    'STRING',
)

literals = ('(', ')', '{', '}', ',')

t_ignore  = ' \t\r'

def t_eol(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_comment(t):
    r'\#[^\n]*\n'
    t.lexer.lineno += 1

def t_KW(t):
    r'UAG|HAG|ASG|RULE|CALC'
    t.type = t.value
    return t

def t_INP(t):
    r'INP[A-L]'
    t.type = 'INP'
    return t

def t_INTEGER(t):
    r'[0-9]+'
    t.type = 'INTEGER'
    t.value = int(t.value)
    return t

def t_bare_STRING(t):
    r'[a-zA-Z0-9_\-+:.\[\]<>;]+'
    t.type = 'STRING'
    return t

def t_quoted_STRING(t):
    r'"([^"\n\\]|\\.)*"'
    t.type = 'STRING'
    t.value = t.value[1:-1] # asLib_lex.l doesn't un-escape  .encode('utf-8').decode('unicode_escape')
    return t

def t_error(t):
    raise ACFError("Illegal char %s at %d"%(repr(t.value[:5]), t.lexer.lineno+1))

if __name__=='__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    lexer = lex.lex(debug=1, optimize=0, debuglog=logging.getLogger(__name__))
    lex.runmain(lexer=lexer)
