
from ..wrapper import Type

# common sub-structs
timeStamp = Type(id='time_t', spec=[
    ('secondsPastEpoch', 'l'),
    ('nanoseconds', 'i'),
    ('userTag', 'i'),
])
alarm = Type(id='alarm_t', spec=[
    ('severity', 'i'),
    ('status', 'i'),
    ('message', 's'),
])
