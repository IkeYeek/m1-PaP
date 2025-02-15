import inspect

def guess_type_and_convert(string):
    if string.lower() == "none":
        return None
    if string.lower() == "true":
        return True
    elif string.lower() == "false":
        return False
    try:
        return int(string)
    except ValueError:
        try:
            return float(string)
        except ValueError:
            return string


def filterParameters(keylist, argList):
    params = {}
    unused = []

    for element in argList:
        try:
            key, value = element.split('=')
            if key in keylist:
                params[key] = guess_type_and_convert(value)
            else:
                unused.append(element)
        except ValueError:
            unused.append(element)
    return params,unused


def scanParameters(fun, argList):
    argspec = inspect.getfullargspec(fun)
    args = argspec.args + argspec.kwonlyargs
    return filterParameters(args, argList)

def updateFunParameters(fun, argList, **kwargs):
    newParamaters, unused =  scanParameters(fun, argList)
    kwargs.update(newParamaters)
    return kwargs, unused

def updateParameters(argList, **kwargs):
    newParamaters, unused =  scanParameters(fun, argList)
    kwargs.update(newParamaters)
    return kwargs, unused


def searchParameter(key, argList, default = None):
    for element in argList:
        if '=' in element:
            akey, value = element.split('=')
            if akey == key:
                return guess_type_and_convert(value)
    return default  
