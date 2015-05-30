# bintools
tools to help process binary data

## bintools.simplebsd

example:

    from bintools import simplebsd
    po = simplebsd.parse_a('00A4040006010203040501')
    with po.struct('APDU'):
        po.b('CLA', 1)
        po.b('INS', 1)
        po.b('P1P2', 2)
        lc = po.u('Lc', 1).int()
        po.b('CDATA', lc)
    print po.visit().format()
output:
> Struct:APDU         : 
>
>   Bytes:CLA           : 00
>
>   Bytes:INS           : A4
>
>   Bytes:P1P2          : 0400
>
>   UInt:Lc             : 06 (6)
>
>   Bytes:CDATA         : 010203040501
>   
