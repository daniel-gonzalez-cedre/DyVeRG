from tqdm import tqdm
import networkx as nx

from baselines.models_dynamic import DyVeRG, GraphRNN


# dynamic DyVeRG
def dyverg_generate(graphs, mu: int = 4, clustering: str = 'leiden', verbose: bool = False) -> list[nx.Graph]:
    """
        Dynamic graph generator.
        Learns a instance of DyVeRG across every gₜ in graphs, initially conditioned on graphs[0].
        Afterwards, generates by sampling DyVeRG at each timestep.
    """
    grammar = DyVeRG(graphs, mu=mu, clustering=clustering, verbose=verbose)
    generated_graphs = [grammar.generate(t, graphs[t].order())
                        for t in tqdm(range(len(graphs)), desc='DyVeRG generating', disable=(not verbose))]

    return generated_graphs


def main():
    pass


if __name__ == '__main__':
    main()
