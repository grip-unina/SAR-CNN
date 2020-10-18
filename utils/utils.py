import argparse
import os
import pickle

def get_exp_dir(basedir):
    imax = -1
    dirs = os.listdir(basedir)
    for d in dirs:
        try:
            imax = max(imax, int(d[3:7]))
        except:
            pass
    imax = imax+1
    return "exp%04d" % imax

def args2obj(args):
    dargs = dict(vars(args))
    ks = list(dargs)
    for k in ks:
        parts = k.split(".")
        if len(parts) > 1:
            o = dargs
            for p in parts[:-1]:
                try:
                    o[p]
                except:
                    o[p] = {}
                o = o[p]
            o[parts[-1]] = dargs[k]
    return argparse.Namespace(**dargs)


def add_commandline_flag(parser, name_true, name_false, default):
    parser.add_argument(name_true , action="store_true" , dest=name_true[2:])
    parser.add_argument(name_false, action="store_false", dest=name_true[2:])
    parser.set_defaults(**{name_true[2:]: default})

def load_args(expdir):
    with open(os.path.join(expdir, "args.pkl"), "rb") as fid:
        return pickle.load(fid)

def save_args(expdir, args):
    with open(os.path.join(expdir, "args.pkl"), "wb") as fid:
        pickle.dump(args,fid)

def get_module_name_dict(root, rootname="/"):
    def _rec(module, d, name):
        for key, child in module.__dict__["_modules"].items():
            d[child] = name + key + "/"
            _rec(child, d, d[child])

    d = {root: rootname}
    _rec(root, d, d[root])
    return d

def parameters_by_module(net, name=""):
    modulenames = get_module_name_dict(net, name + "/")
    params = [{"params": p, "name": n, "module": modulenames[m]} for m in net.modules() for n,p in m._parameters.items() if p is not None]
    return params

def parameter_count(net):
    parameters = net.parameters()
    nparams = 0
    for p in parameters:
        nparams += p.data.numel()

    return nparams





