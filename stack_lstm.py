import dynet

from layer import add_layer, combine_layers, tanh, sigmoid, linear

START_SYMBOL, SEPARATOR_SYMBOL, END_SYMBOL = range(3)

class NeuralStack:

    def __init__(self, embedding_size, batch_size):
        self.vector_zero = dynet.zeroes((embedding_size,), batch_size)
        self.scalar_zero = dynet.zeroes((1,), batch_size)
        self.reading_depth = dynet.inputTensor([1.0] * batch_size, True)
        self.elements = []

    def reading(self):
        terms = [self.vector_zero]
        strength_left = self.reading_depth
        for element in reversed(self.elements):
            terms.append(element.value * dynet.bmin(
                element.strength,
                dynet.bmax(self.scalar_zero, strength_left)))
            strength_left -= element.strength
        return dynet.esum(terms)

    def push(self, strength, value):
        self.elements.append(NeuralStack.Element(value, strength))

    def pop(self, strength):
        strength_left = strength
        for element in reversed(self.elements):
            old_strength = element.strength
            element.strength = dynet.bmax(
                self.scalar_zero,
                old_strength - dynet.bmax(
                    self.scalar_zero,
                    strength_left))
            strength_left -= old_strength

    class Element:

        def __init__(self, value, strength):
            self.value = value
            self.strength = strength

class StackLSTMBuilder:

    INPUT_MODE, OUTPUT_MODE = range(2)

    def __init__(self, params, source_alphabet_size, embedding_size, hidden_units,
            stack_embedding_size):
        input_size = source_alphabet_size + 2
        output_size = source_alphabet_size + 1
        self.stack_embedding_size = stack_embedding_size
        self.input_embeddings = params.add_lookup_parameters(
            (input_size, embedding_size),
            name=b'input-embeddings')
        self.output_embeddings = params.add_lookup_parameters(
            (output_size, embedding_size),
            name=b'output-embeddings')
        self.controller = dynet.CoupledLSTMBuilder(
            1, embedding_size + stack_embedding_size, hidden_units, params)
        # Intentionally set the gain for the sigmoid layers low, since this
        # seems to work better
        gain = 0.5
        self.pop_strength_layer = add_layer(
            params, hidden_units, 1, sigmoid,
            weights_initializer=dynet.GlorotInitializer(False, gain),
            # Initialize the pop bias to -1 to allow information to propagate
            # through the stack
            bias_initializer=dynet.ConstInitializer(-1.0),
            name='pop-strength')
        self.push_strength_layer = add_layer(
            params, hidden_units, 1, sigmoid,
            weights_initializer=dynet.GlorotInitializer(False, gain),
            bias_initializer=dynet.GlorotInitializer(False, gain),
            name='push-strength')
        self.push_value_layer = add_layer(
            params, hidden_units, stack_embedding_size, tanh, name='push-value')
        self.output_layer = combine_layers([
            add_layer(params, hidden_units, hidden_units, tanh, name='output'),
            # This adds an extra affine layer between the tanh and the softmax
            add_layer(params, hidden_units, output_size, linear, name='softmax')
        ])

    def initial_state(self, batch_size):
        return StackLSTMBuilder.State(
            StackLSTMBuilder.Run(
                self, NeuralStack(self.stack_embedding_size, batch_size)),
            self.controller.initial_state())

    class State:

        def __init__(self, run, controller_state):
            self.run = run
            self.controller_state = controller_state

        def next(self, index_batch, mode):
            if mode == StackLSTMBuilder.INPUT_MODE:
                embeddings = self.run.builder.input_embeddings
            else:
                embeddings = self.run.builder.output_embeddings
            embedding = dynet.lookup_batch(embeddings, index_batch)
            stack_reading = self.run.stack.reading()
            controller_input = dynet.concatenate([embedding, stack_reading])
            controller_state = self.controller_state.add_input(controller_input)
            controller_output = controller_state.output()
            pop_strength = self.run.pop_strength_layer(controller_output)
            push_strength = self.run.push_strength_layer(controller_output)
            push_value = self.run.push_value_layer(controller_output)
            self.run.stack.pop(pop_strength)
            self.run.stack.push(push_strength, push_value)
            return StackLSTMBuilder.State(self.run, controller_state)

        def output(self):
            return self.run.output_layer(self.controller_state.output())

    class Run:

        def __init__(self, builder, stack):
            self.builder = builder
            self.stack = stack
            self.pop_strength_layer = builder.pop_strength_layer.expression()
            self.push_strength_layer = builder.push_strength_layer.expression()
            self.push_value_layer = builder.push_value_layer.expression()
            self.output_layer = builder.output_layer.expression()

def input_symbol_to_index(symbol):
    return symbol - (symbol > END_SYMBOL)

def output_symbol_to_index(symbol):
    return symbol - 2

def output_index_to_symbol(index):
    return index + 2
