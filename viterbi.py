class Node:
    def __init__(self, state: str, probability: float, back_pointer):
        self.back_pointer: Node = back_pointer
        self.state: str = state
        self.probability: float = probability


def keys_match(dict_a: dict, dict_b: dict) -> bool:
    return dict_a.keys() == dict_b.keys()


def key_with_max_val(d: dict) -> str:
    """https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def node_with_max_prob(d: dict) -> Node:
    max_node = Node(None, 0.0, None)
    for node in d.values():
        if node.probability > max_node.probability:
            max_node = node
    return max_node


class Viterbi:
    def __init__(self, initial_probabilities: dict, emission_probabilities: dict, transition_probabilities: dict):
        assert keys_match(initial_probabilities, emission_probabilities) and\
               keys_match(initial_probabilities, transition_probabilities), "Hidden states must be consistent!"
        self.initial = initial_probabilities
        self.emission = emission_probabilities
        self.transitions = transition_probabilities

    def predict_path(self, observations: list) -> list:
        matrix: list = [{}]

        for state in self.initial:
            matrix[0][state] = Node(state, self.initial[state]*self.emission[state][observations[0]], None)

        # fill initial probabilities
        for prev_idx, observation in enumerate(observations[1:]):
            matrix.append({})
            for state in self.transitions:
                transitions: dict = {}
                for prev_state in matrix[prev_idx]:
                    prev_prob = matrix[prev_idx][prev_state].probability
                    transition_prob = self.transitions[prev_state][state]*prev_prob
                    transitions[prev_state] = transition_prob
                last_state = key_with_max_val(transitions)
                probability = self.emission[state][observation]*transitions[last_state]
                matrix[prev_idx+1][state] = Node(state, probability, matrix[prev_idx][last_state])

        current_node: Node = node_with_max_prob(matrix[-1])
        sequence: list = []
        while current_node is not None:
            sequence.insert(0, current_node.state)
            current_node = current_node.back_pointer

        return sequence

