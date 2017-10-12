UNIPOLAR = 'unipolar'
BIPOLAR = 'bipolar'


def unipolar(x):
    return 1 if x >= 0 else 0


def bipolar(x):
    return 1 if x >= 0 else -1


def get_activation(activation):
    return ACTIVATION_FUNCTIONS[activation]

ACTIVATION_FUNCTIONS = {
    UNIPOLAR: unipolar,
    BIPOLAR: bipolar
}
