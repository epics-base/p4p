#!/usr/bin/env python

from __future__ import print_function

import zipfile
import tempfile
import shutil
import os
import subprocess as SP

class TempDir(object):
    def __init__(self):
        self.name = None
    def close(self):
        if self.name is not None:
            shutil.rmtree(self.name)
            self.name = None
    def __enter__(self):
        self.name = tempfile.mkdtemp()
        return self.name
    def __exit__(self, A, B, C):
        self.close()

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()

    P.add_argument('--tag')
    P.add_argument('--rm', action='store_true', default=False)
    P.add_argument('wheel')

    return P.parse_args()

def main(args):
    wheelname, wheelext = os.path.splitext(os.path.basename(args.wheel))
    wheeldir = os.path.dirname(args.wheel)

    parts = wheelname.split('-')
    assert len(parts)==5, parts
    name, version, tag, abi, platname = parts

    target = '-'.join((tag, abi, platname))

    print('For', name, version)
    print('replace', target, 'with', args.tag)

    with TempDir() as temp:
        print("Working in", temp)
        infodir = os.path.join(temp, '%s-%s.dist-info'%(name, version))

        with zipfile.ZipFile(args.wheel, 'r') as I:
            I.extractall(temp)

        SP.check_call('ls %s'%temp, shell=True)

        print('Old WHEEL')
        wheelinfo = []
        with open(os.path.join(infodir, 'WHEEL'), 'r') as F:
            for line in F:
                print(repr(line))
                if line=='Root-Is-Purelib: true\n':
                    line='Root-Is-Purelib: false\n'
                elif line.startswith('Tag: '):
                    line = 'Tag: %s\n'%args.tag
                wheelinfo.append(line.strip())

        print('New WHEEL')
        with open(os.path.join(infodir, 'WHEEL'), 'wb') as F:
            for line in wheelinfo:
                print(line)
                F.write(line.encode('utf-8'))
                F.write(b'\r\n')

        newname = '%s-%s-%s%s'%(name, version, args.tag, wheelext)
        outfile = os.path.join(wheeldir, newname)

        with zipfile.ZipFile(outfile, 'w', zipfile.ZIP_DEFLATED) as F:
            for root, dirs, files in os.walk(temp):
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, temp)
                    print('->', full)
                    F.write(full, rel)

        if args.rm:
            os.remove(args.wheel)

if __name__=='__main__':
    main(getargs())
