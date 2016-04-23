import theano

class DisconnectedGrad(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [theano.gradient.DisconnectedType()() for g_out in g_outs]

    def connection_pattern(self, node):
        return [[False]]

disconnected_grad = DisconnectedGrad()
