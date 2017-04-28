
class Scope:
    """Creates and stores namespaces used by modules.
    
    Each module has a scope that is used to store global context about the 
    network. A module can create new namespaces or access existing ones.
    The objective is to provide a namespace that a module can use to store 
    and share information across modules. It also provides unique module 
    identifiers.

    """

    def __init__(self):
        self.s = {}

    def register_namespace(self, name):
        if name in self.s:
            return ValueError
        else:
            self.s[name] = {}

    def get_namespace(self, name):
        return self.s[name]

    def get_valid_name(self, prefix):
        n = 0
        while True:
            name = prefix + '-' + str(n)
            if name not in self.s:
                return name
            n += 1

