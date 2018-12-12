'''
Print the dimensionality of an embedding file to stdout
'''
import sys
from . import *
from .common import *

if __name__=='__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog EMBF')
        parser.add_option('-d', '--dim-only', dest='dim_only',
            action='store_true', default=False,
            help='Only print out dimensionality')
        parser.add_option('--format', dest='format',
            type='choice', choices=list(CLI_Formats.options()), default=CLI_Formats.default(),
            help='current format of EMBF')
        (options, args) = parser.parse_args()
        if len(args) != 1:
            parser.print_help()
            exit()
        return args[0], options
    embf, options = _cli()
    fmt, mode = CLI_Formats.parse(options.format)

    (nwords, dim) = getSize(embf, format=fmt, mode=mode)
    if options.dim_only:
        sys.stdout.write('%d\n' % dim)
    else:
        sys.stdout.write('%d %d\n' % (nwords, dim))
