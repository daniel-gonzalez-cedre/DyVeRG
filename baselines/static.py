from tqdm import tqdm
import networkx as nx

from baselines.models_static import VeRG, uniform, erdos_renyi, chung_lu
from baselines.graphrnn.fit import fit
from baselines.graphrnn.gen import gen


# static DyVeRG
def VeRG_generate(graphs, mu: int = 4, clustering: str = 'leiden', verbose: bool = False) -> list[nx.Graph]:
    """
        Static graph generator.
        For each gₜ in graphs, generates from an instance of DyVeRG learned independently at time t.
    """
    generated_graphs = []
    for t, g in tqdm(enumerate(graphs), desc='VRG generating', total=len(graphs), disable=(not verbose)):
        grammar = VeRG(g, time=t, mu=mu, clustering=clustering)
        generated = grammar.generate(t, g.order())
        generated_graphs.append(generated)

    return generated_graphs


# TODO
# static CNRG
def cnrg_generate(graphs, mu: int = 4, clustering: str = 'leiden', verbose: bool = False) -> list[nx.Graph]:
    raise NotImplementedError


# TODO
# static HRG
def hrg_generate(graphs, mu: int = 4, clustering: str = 'leiden', verbose: bool = False) -> list[nx.Graph]:
    raise NotImplementedError


def graphrnn_generate(graphs) -> list[nx.Graph]:
    args, model, output = fit(graphs)
    generated_graphs = gen(args=args, model=model, output=output)
    return generated_graphs


# static random (chooses uniformly from all graphs on nₜ nodes and mₜ edges)
def uniform_generate(graphs, verbose: bool = False) -> list[nx.Graph]:
    """
        Static graph generator.
        For each gₜ in graphs, selects uniformly at random from the graphs on |V(gₜ)| nodes and |E(gₜ)| edges.
    """
    generated_graphs = [uniform(graph)
                        for graph in tqdm(graphs, desc='random generating', total=len(graphs), disable=(not verbose))]
    return generated_graphs


# static erdos-renyi
def er_generate(graphs, verbose: bool = False) -> list[nx.Graph]:
    """
        Static graph generator.
        For each gₜ in graphs, learns an Erdos-Renyi random model on gₜ and samples a graph from it.
    """
    generated_graphs = [erdos_renyi(graph)
                        for graph in tqdm(graphs, desc='Erdos-Renyi generating', total=len(graphs), disable=(not verbose))]
    return generated_graphs


# static Chung-Lu
def cl_generate(graphs, verbose: bool = False) -> list[nx.Graph]:
    """
        Static graph generator.
        For each gₜ in graphs, learns a Chung-Lu random model on gₜ and samples a graph from it.
        The ``switch`` parameter selects the implementation of Chung-Lu.
    """
    # if False:
    #     if False:
    #         generated_graphs = [chung_lu_switch(graph)
    #                             for graph in tqdm(graphs, desc='Chung-Lu generating', total=len(graphs), disable=(not verbose))]
    #     else:
    #         generated_graphs = [chung_lu(graph)
    #                             for graph in tqdm(graphs, desc='Chung-Lu generating', total=len(graphs), disable=(not verbose))]

    generated_graphs = [chung_lu(graph)
                        for graph in tqdm(graphs, desc='Chung-Lu generating', total=len(graphs), disable=(not verbose))]
    return generated_graphs


def main():
    pass


if __name__ == '__main__':
    main()
