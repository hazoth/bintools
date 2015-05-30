import copy

def cons(a, b):
    return (a, b)
def car(a):
    return a[0]
def cdr(a):
    return a[1]
def setcar(a, b):
    return (b, a[1])

class ReaderError(Exception):pass
class Reader(object):
    def __init__(self, s, p = 0):
        self.raw = s
        self.pos = p
        self.abs_limit = len(s)
        self.limit_stack = ()

    def clone(self):
        return copy.copy(self)

    def peek(self, size):
        p = self.pos + size
        if p > self.abs_limit:
            raise ReaderError('%d > %d'%(p, self.abs_limit))
        return self.raw[self.pos:p]

    def read(self, size):
        p = self.pos + size
        if p > self.abs_limit:
            raise ReaderError('%d > %d'%(p, self.abs_limit))
        value = self.raw[self.pos:p]
        self.pos = p
        return value

    @property
    def limit(self):
        return max(self.abs_limit - self.pos, 0)

    def set_limit(self, value):
        p = self.pos + value
        if p > self.abs_limit or p < self.pos:
            raise ReaderError('%d > %d'%(p, self.abs_limit))
        self.abs_limit = p

    def assert_limit(self, value):
        if self.abs_limit !=  self.pos + value:
            raise ReaderError('%d != %d + %d'%(self.abs_limit, self.pos, value))
        return

    def push_limit(self, value):
        self.limit_stack = cons(self.abs_limit, self.limit_stack)
        self.set_limit(value)

    def pop_limit(self):
        self.abs_limit = car(self.limit_stack)
        self.limit_stack = cdr(self.limit_stack)


from collections import namedtuple

TreeNode = namedtuple('TreeNode', 'item, sibling, child, ')
HeadNode = namedtuple('HeadNode', 'item, sibling, child, parent')
class ImmutableTree(object):
    def __init__(self):
        self._head = HeadNode(None, None, None, None)

    def clone(self):
        return copy.copy(self)

    def append(self, item):
        x = TreeNode(*self._head[:3])
        self._head = HeadNode(item, x, None, self._head[3])

    def getcwd(self):
        return self._head.parent

    def pushd(self, item):
        self.append(item)
        self._head = HeadNode(item, None, None, self._head)

    def popd(self):
        x = TreeNode(*self._head[:3])
        y = self._head.parent
        self._head = HeadNode(y.item, y.sibling, x, y.parent)

    def reversed_children(self, node):
        if type(node) is TreeNode:
            i = node.child
            if i is None:
                return
        else:
            i = self._head
            if i is node:
                if i.child is None:
                    return
                i = i.child
            else:
                while i.parent is not node:
                    i = i.parent
        while i.sibling is not None:
            yield i
            i = i.sibling
        return

    def get_children(self, node):
        lst = list(self.reversed_children(node))
        lst.reverse()
        return lst

    def count_children(self, node):
        return sum(1 for i in self.reversed_children(node))

    def format(self, node = None, indent = '|- '):
        if node is None:
            lst = []
        else:
            lst = ['%s%s'%(indent, node.item.__class__.__name__)]
            indent = '  ' + indent
        for i in self.get_children(node):
            lst.append(self.format(i, indent))
        return '\n'.join(lst)

from binascii import hexlify
b2a = lambda x:hexlify(x).upper()
b2i = lambda x:int(hexlify(x), 16)

def a2si(x):
    m = 16 ** len(x)
    x = int(x, 16)
    if x >= (m >> 1):
        x -= m
    return x
b2si = lambda x:a2si(hexlify(x))

class DataItem(object):
    class Bytes(object):
        __slots__ = ('name', 'raw')
        def __init__(self, name, value):
            self.name = name
            self.raw = value

        def hex(self):
            return b2a(self.raw)

    class UInt(Bytes):
        __slots__ = ('name', 'raw')
        def int(self):
            return b2i(self.raw)

    class SInt(Bytes):
        __slots__ = ('name', 'raw')
        def int(self):
            return b2si(self.raw)

    class Utf8(Bytes):
        __slots__ = ('name', 'raw')
        @property
        def value(self):
            return self.raw.decode('utf8')

        def int(self):
            return int(self.raw)

    class BerTag(Bytes):
        __slots__ = ('name', 'raw')
        @classmethod
        def parse(self, reader, name):
            lst = [reader.read(1)]
            if (ord(lst[-1]) & 0x1F) == 0x1F:
                lst.append(reader.read(1))
                while ord(lst[-1]) >= 0x80:
                    lst.append(reader.read(1))
            return self(name, ''.join(lst))

        @property
        def c_p(self):
            return ((ord(self.raw[0]) >> 5) & 1) # c = 1, p = 0

    class BerLength(Bytes):
        __slots__ = ('name', 'raw')
        @classmethod
        def parse(self, reader, name):
            v = reader.read(1)
            if ord(v) > 128:
                v += reader.read(ord(v) & 0x7F)
            return self(name, v)

        def int(self):
            if len(self.raw) > 1:
                return b2i(self.raw[1:])
            else:
                return ord(self.raw[0])

    class Struct(object):
        __slots__ = ('name',)
        def __init__(self, name):
            self.name = name    

    class Array(Struct):
        __slots__ = ('name',)

    class BerTLV(Struct):
        __slots__ = ('name',)


class DataVisitor(object):
    def __init__(self, runtime, node = None, exit_hook = None):
        assert isinstance(runtime, Runtime)
        self.runtime = runtime
        self.node = node
        self.exit_hook = exit_hook

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.exit_hook is not None:
            self.exit_hook()
            self.exit_hook = None
        return
    close = __exit__

    def __getitem__(self, index):
        tree = self.runtime.result
        node = self.node
        if type(index) is not tuple:
            index = (index,)
        for i in index:
            if isinstance(i, basestring):
                for node in tree.reversed_children(node):
                    if node.item.name == i:
                        break
                else:
                    raise KeyError
            elif isinstance(i, int):
                node = tree.get_children(node)[i]
            else:
                raise NotImplementedError
        if isinstance(node.item, DataItem.Struct):
            return DataVisitor(self.runtime, node)
        return node.item

    def to_lines(self, node = None, indent = '', index = None):
        lst = []
        if node is not None:
            item = node.item
            typename = item.__class__.__name__
            name = item.name or ''
            index = index is not None and '[%d] '%index or ''
            value = getattr(self, 'format_' + typename)(item)
            prefix = '{}{}:{}'.format(index, typename, name)
            line = '{}{:<20s}: {}'.format(indent, prefix, value)
            lst.append(line)
            indent += '  '
        else:
            item = None
        children = self.runtime.result.get_children(node)
        if isinstance(item, DataItem.Array):
            for index, i in enumerate(children):
                lst.extend(self.to_lines(i, indent, index))
        else:
            for i in children:
                lst.extend(self.to_lines(i, indent))
        return lst

    def format(self, node = None, indent = ''):
        return '\n'.join(self.to_lines(node, indent))

    def format_Bytes(self, item):
        return '%s'%(item.hex())

    def format_UInt(self, item):
        return '%s (%d)'%(item.hex(), item.int())

    def format_SInt(self, item):
        return '%s (%d)'%(item.hex(), item.int())

    def format_Utf8(self, item):
        return '"%s"'%(item.raw)

    format_BerTag = format_Bytes
    format_BerLength = format_SInt

    def format_Struct(self, item):
        return ''

    format_Array = format_Struct
    format_BerTLV = format_Struct

class Runtime(object):
    # class Frame(object):
    #     def __init__(self, **kwargs):
    #         self.__dict__.update(kwargs)

    #     def clone(self):
    #         return copy.copy(self)

    #     def frozen(self):
    #         return self.__dict__.items()

    _fields = [
        'reader', 
        'result',
        # 'frame',
    ]
    def __init__(self, reader):
        assert isinstance(reader, Reader)
        self.reader = reader
        self.result = ImmutableTree()
        # self.frame = self.Frame()
        self.savepoints = []

    def save(self):
        sp = [getattr(self, i).clone() for i in self._fields]
        self.savepoints.append(sp)
        return sp

    def rollback(self, sp = None):
        if sp is None:
            sp = self.savepoints.pop()
        else:
            while self.savepoints:
                if self.savepoints.pop() is sp:
                    break
        for i, j in zip(self._fields, sp):
            setattr(self, i, j)
        return

    def release(self):
        return self.savepoints.pop()

    # def run(runtime, runner):
    #     runner = runner.setup(runtime)
    #     e = None
    #     while True:
    #         try:
    #             if e is not None:
    #                 runner = runner.handle_exception(runtime, e)
    #             while runner:
    #                 runner = runner.run(runtime)
    #         except Exception as e:
    #             runner = runtime.rollback(e)
    #             continue
    #         break
    #     return

    def visit(self, node = None, *args, **kwargs):
        return DataVisitor(self, node, *args, **kwargs)

    def struct(self, name = None, size_limit = None, Construct = DataItem.Struct):
        item = Construct(name)
        self.result.pushd(item)
        if size_limit is not None:
            self.reader.push_limit(size_limit)
        def exit_hook():
            if size_limit is not None:
                self.reader.pop_limit()
            self.result.popd()
        return self.visit(self.result.getcwd(), exit_hook = exit_hook)

    def array(self, name = None, size_limit = None):
        return self.struct(
            name = name, 
            size_limit = size_limit, 
            Construct = DataItem.Array,
        )

    def b(self, name = None, size = None, Construct = DataItem.Bytes):
        if size is None:
            if isinstance(name, int):
                size, name = name, None
            else:
                size = 1
        value = self.reader.read(size)
        item = Construct(name, value)
        self.result.append(item)
        return item

    def u(self, name = None, size = None):
        return self.b(name, size, DataItem.UInt)

    def s(self, name = None, size = None):
        return self.b(name, size, DataItem.SInt)

    def c(self, name = None, size = None):
        return self.b(name, size, DataItem.Utf8)

    def berTag(self, name = None):
        item = DataItem.BerTag.parse(self.reader, name)
        self.result.append(item) 
        return item

    def berLength(self, name = None):
        item = DataItem.BerLength.parse(self.reader, name)
        self.result.append(item) 
        return item

    def berTLV_struct(self, name = None):
        visitor = self.struct(
            name = name, 
            size_limit = self.reader.limit, 
            Construct = DataItem.BerTLV,
        )
        self.berTag('T')
        length = self.berLength('L').int()
        self.reader.set_limit(length)
        return visitor

    def berTLV(self, name = None, recursive = False):
        value_name = 'V'
        with self.berTLV_struct(value_name) as v0:
            if recursive and v0[0].c_p:
                v1 = self.array(value_name)
                stack = []
                while True:
                    while self.reader.limit:
                        v = self.berTLV_struct()
                        if v[0].c_p:
                            stack.append(v)
                            stack.append(self.array(value_name))
                            continue
                        self.b(value_name, self.reader.limit)
                        v.close()
                    if not stack:
                        break
                    stack.pop().close()
                    stack.pop().close()
                v1.close()
            else:
                self.b(value_name, self.reader.limit)
        return v0

def parse_b(s, start = 0):
    return Runtime(Reader(s, start))

def parse_a(s, start = 0):
    return parse_b(s.decode('hex'))


if __name__ == '__main__':
    # test 1
    po = parse_a('00A4040006010203040501')
    with po.struct('APDU'):
        po.b('CLA', 1)
        po.b('INS', 1)
        po.b('P1P2', 2)
        lc = po.u('Lc', 1).int()
        po.b('CDATA', lc)
        po.b('Le', 0)
    print po.visit().format()
    print ''
    # test 2
    po = parse_a('FF4024E222E1184F00C114DD9287A6192790E235DB3533FABED9B513521DFCE306D00101D10101')
    po.berTLV('ARA Rule', recursive = True)
    print po.visit().format()
