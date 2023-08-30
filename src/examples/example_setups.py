from networkx.drawing import nx_agraph
from networkx.classes.function import create_empty_copy
import pygraphviz
from src.utils.sem_utils.toy_sems import StationaryDependentSEM, StationaryIndependentSEM, NonStationaryDependentSEM, HealthcareSEM, EpidemiologySEM, ExtendedEpidemiologySEM
from src.utils.dag_utils.graph_functions import make_graphical_model
from src.experimental.experiments import optimal_sequence_of_interventions
from src.utils.sequential_intervention_functions import get_interventional_grids
from src.utils.utilities import powerset
from copy import deepcopy

def setup_stat_scm_ceo_extended_epidem(random_state, n_anchor, T: int = 0):

    # Specific choice
    SEM = ExtendedEpidemiologySEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    # For CEO we create a list of graphs
    graphs = []

    base_dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["U", "T", "L", "R", "Y", "P", "M", "E", "Z", "X"], verbose=True
    )
    base_dag = nx_agraph.from_agraph(pygraphviz.AGraph(base_dag_view))

    true_dag = create_empty_copy(base_dag)

    true_dag.add_edge('U_{}'.format(0), 'L_{}'.format(0))
    true_dag.add_edge('U_{}'.format(0), 'Y_{}'.format(0))

    true_dag.add_edge('T_{}'.format(0), 'L_{}'.format(0))
    true_dag.add_edge('T_{}'.format(0), 'Y_{}'.format(0))
    true_dag.add_edge('T_{}'.format(0), 'R_{}'.format(0))

    true_dag.add_edge('L_{}'.format(0), 'R_{}'.format(0))

    true_dag.add_edge('R_{}'.format(0), 'Y_{}'.format(0))
    # Added confounders
    true_dag.add_edge('P_{}'.format(0), 'U_{}'.format(0))
    true_dag.add_edge('M_{}'.format(0), 'T_{}'.format(0))
    true_dag.add_edge('E_{}'.format(0), 'L_{}'.format(0))
    true_dag.add_edge('Z_{}'.format(0), 'R_{}'.format(0))
    true_dag.add_edge('X_{}'.format(0), 'Y_{}'.format(0))

    # Wrong graph 1
    wrong_dag_1 = deepcopy(true_dag)
    wrong_dag_1.remove_edge(u="T_0",v="Y_0")

    #TODO
    wrong_dag_1.T = 1
    graphs.append(wrong_dag_1)

    # Wrong graph 2
    wrong_dag_2 = deepcopy(true_dag)
    wrong_dag_2.remove_edge(u="L_0",v="R_0")
    wrong_dag_2.remove_edge(u="T_0",v="R_0")

    #TODO
    wrong_dag_2.T = 1
    graphs.append(wrong_dag_2)

    # Wrong graph 3
    wrong_dag_3 = deepcopy(true_dag)
    wrong_dag_3.remove_edge(u="L_0",v="R_0")
    wrong_dag_3.remove_edge(u="T_0",v="Y_0")
    wrong_dag_3.add_edge('L_{}'.format(0), 'Y_{}'.format(0))
    wrong_dag_3.add_edge('R_{}'.format(0), 'L_{}'.format(0))

    #TODO
    wrong_dag_3.T = 1
    graphs.append(wrong_dag_3)



    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = [ ("T",), ("R",), ("T", "R")]
    # Specify the intervention domain for each variable
    intervention_domain = {"T": [0, 4], "R": [0, 4]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=n_anchor)

    _, _, true_objective_values, _, _, all_CE = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=true_dag,
        T=T,
        model_variables=["U", "T", "L", "R", "Y", "P", "M", "E", "Z", "X"],
        target_variable="Y",
        random_state=random_state
    )

    return init_sem, sem, base_dag_view, graphs, exploration_sets, intervention_domain, true_objective_values, all_CE


def setup_stat_scm_ceo_epidem(random_state, n_anchor, T: int = 0):

    # Specific choice
    SEM = EpidemiologySEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    # For CEO we create a list of graphs
    graphs = []

    base_dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["U", "T", "L", "R", "Y"], verbose=True
    )
    base_dag = nx_agraph.from_agraph(pygraphviz.AGraph(base_dag_view))

    true_dag = create_empty_copy(base_dag)

    true_dag.add_edge('U_{}'.format(0), 'L_{}'.format(0))
    true_dag.add_edge('U_{}'.format(0), 'Y_{}'.format(0))

    true_dag.add_edge('T_{}'.format(0), 'L_{}'.format(0))
    true_dag.add_edge('T_{}'.format(0), 'Y_{}'.format(0))
    true_dag.add_edge('T_{}'.format(0), 'R_{}'.format(0))

    true_dag.add_edge('L_{}'.format(0), 'R_{}'.format(0))

    true_dag.add_edge('R_{}'.format(0), 'Y_{}'.format(0))

    # Wrong graph 1
    wrong_dag_1 = deepcopy(true_dag)
    wrong_dag_1.remove_edge(u="T_0",v="Y_0")

    #TODO
    wrong_dag_1.T = 1
    graphs.append(wrong_dag_1)

    # Wrong graph 2
    wrong_dag_2 = deepcopy(true_dag)
    wrong_dag_2.remove_edge(u="L_0",v="R_0")
    wrong_dag_2.remove_edge(u="T_0",v="R_0")

    #TODO
    wrong_dag_2.T = 1
    graphs.append(wrong_dag_2)

    # Wrong graph 3
    wrong_dag_3 = deepcopy(true_dag)
    wrong_dag_3.remove_edge(u="L_0",v="R_0")
    wrong_dag_3.remove_edge(u="T_0",v="Y_0")
    wrong_dag_3.add_edge('L_{}'.format(0), 'Y_{}'.format(0))
    wrong_dag_3.add_edge('R_{}'.format(0), 'L_{}'.format(0))

    #TODO
    wrong_dag_3.T = 1
    graphs.append(wrong_dag_3)



    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = [ ("T",), ("R",), ("T", "R")]
    # Specify the intervention domain for each variable
    intervention_domain = {"T": [0, 4], "R": [0, 4]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=n_anchor)

    _, _, true_objective_values, _, _, all_CE = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=true_dag,
        T=T,
        model_variables=["U", "T", "L", "R", "Y"],
        target_variable="Y",
        random_state=random_state
    )

    return init_sem, sem, base_dag_view, graphs, exploration_sets, intervention_domain, true_objective_values, all_CE


def setup_stat_scm_ceo_healthcare(random_state, n_anchor, T: int = 0):

    # Specific choice
    SEM = HealthcareSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    # For CEO we create a list of graphs
    graphs = []

    base_dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["A","B","R","S","C","Y"], verbose=True
    )  # Base topology that we build on
    base_dag = nx_agraph.from_agraph(pygraphviz.AGraph(base_dag_view))
    base_dag = create_empty_copy(base_dag)

    base_dag.add_edge('A_{}'.format(0), 'B_{}'.format(0))
    base_dag.add_edge('A_{}'.format(0), 'C_{}'.format(0))
    base_dag.add_edge('A_{}'.format(0), 'S_{}'.format(0))
    base_dag.add_edge('A_{}'.format(0), 'R_{}'.format(0))
    base_dag.add_edge('A_{}'.format(0), 'Y_{}'.format(0))

    base_dag.add_edge('B_{}'.format(0), 'S_{}'.format(0))
    base_dag.add_edge('B_{}'.format(0), 'C_{}'.format(0))
    base_dag.add_edge('B_{}'.format(0), 'R_{}'.format(0))
    base_dag.add_edge('B_{}'.format(0), 'Y_{}'.format(0))

    base_dag.add_edge('R_{}'.format(0), 'Y_{}'.format(0))
    base_dag.add_edge('R_{}'.format(0), 'C_{}'.format(0))

    base_dag.add_edge('S_{}'.format(0), 'Y_{}'.format(0))
    base_dag.add_edge('S_{}'.format(0), 'C_{}'.format(0))

    base_dag.add_edge('C_{}'.format(0), 'Y_{}'.format(0))

    true_dag = deepcopy(base_dag)

    graphs.append(true_dag)


    # Wrong graph 1
    wrong_dag_1 = deepcopy(base_dag)
    wrong_dag_1.remove_edge(u="S_0",v="Y_0")
    wrong_dag_1.remove_edge(u="S_0",v="C_0")

    #TODO
    wrong_dag_1.T = 1
    graphs.append(wrong_dag_1)

    # Wrong graph 2
    wrong_dag_2 = deepcopy(base_dag)
    wrong_dag_2.remove_edge(u="R_0",v="Y_0")
    wrong_dag_2.remove_edge(u="R_0",v="C_0")

    wrong_dag_2.T = 1
    graphs.append(wrong_dag_2)

    # Wrong graph 3
    wrong_dag_3 = deepcopy(base_dag)
    wrong_dag_3.remove_edge(u="S_0",v="Y_0")
    wrong_dag_3.remove_edge(u="S_0",v="C_0")
    wrong_dag_3.remove_edge(u="R_0",v="Y_0")
    wrong_dag_3.remove_edge(u="R_0",v="C_0")

    wrong_dag_3.T = 1
    graphs.append(wrong_dag_3)

    # Wrong graph 4
    wrong_dag_4 = deepcopy(base_dag)
    wrong_dag_4.remove_edge(u="A_0",v="Y_0")
    wrong_dag_4.remove_edge(u="A_0",v="C_0")
    wrong_dag_4.remove_edge(u="B_0",v="Y_0")
    wrong_dag_4.remove_edge(u="B_0",v="C_0")

    wrong_dag_4.T = 1
    graphs.append(wrong_dag_4)


    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = [ ("R",),("S",), ("S","R") ]
    # Specify the intervention domain for each variable
    intervention_domain = {"R": [0, 1], "S": [0, 1]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=n_anchor)

    _, _, true_objective_values, _, _, all_CE = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=true_dag,
        T=T,
        model_variables=["A","B","R","S","C","Y"],
        target_variable="Y",
        random_state=random_state
    )

    return init_sem, sem, base_dag_view, graphs, exploration_sets, intervention_domain, true_objective_values, all_CE


def setup_stat_scm_ceo_toy(random_state, n_anchor, T: int = 0):

    # Specific choice
    SEM = StationaryDependentSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    # For CEO we create a list of graphs
    graphs = []

    base_dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    base_dag = nx_agraph.from_agraph(pygraphviz.AGraph(base_dag_view))
    # True is X -> Z -> Y
    true_dag = deepcopy(base_dag)
    graphs.append(true_dag)

    # Wrong graph: Z <- X -> Y
    wrong_dag_1 = deepcopy(base_dag)
    wrong_dag_1.remove_edge(u="Z_0",v="Y_0")
    wrong_dag_1.add_edge(u_for_edge="X_0",v_for_edge="Y_0")
    #TODO
    wrong_dag_1.T = 1
    graphs.append(wrong_dag_1)

    # Wrong graph: X -> Z, Z ->Y, X->Y
    wrong_dag_2 = deepcopy(base_dag)
    wrong_dag_2.add_edge(u_for_edge="X_0",v_for_edge="Y_0")
    wrong_dag_2.T = 1
    graphs.append(wrong_dag_2)

    # Wrong graph: X <- Z -> Y
    wrong_dag_3 = create_empty_copy(base_dag)
    wrong_dag_3.add_edge(u_for_edge="Z_0",v_for_edge="Y_0")
    wrong_dag_3.add_edge(u_for_edge="Z_0",v_for_edge="X_0")
    #TODO
    wrong_dag_3.T = 1
    graphs.append(wrong_dag_3)

    # Wrong graph: X -> Y <- Z
    wrong_dag_4 = create_empty_copy(base_dag)
    wrong_dag_4.add_edge(u_for_edge="Z_0",v_for_edge="Y_0")
    wrong_dag_4.add_edge(u_for_edge="X_0",v_for_edge="Y_0")
    #TODO
    wrong_dag_4.T = 1
    graphs.append(wrong_dag_4)

    # Wrong graph: Z -> X , X -> Y, Z -> Y
    wrong_dag_5 = create_empty_copy(base_dag)
    wrong_dag_5.add_edge(u_for_edge="Z_0",v_for_edge="X_0")
    wrong_dag_5.add_edge(u_for_edge="X_0",v_for_edge="Y_0")
    wrong_dag_5.add_edge(u_for_edge="Z_0",v_for_edge="Y_0")
    #TODO
    wrong_dag_5.T = 1
    graphs.append(wrong_dag_5)

    # Wrong graph: Z -> X -> Y
    wrong_dag_6 = create_empty_copy(base_dag)
    wrong_dag_6.add_edge(u_for_edge="Z_0",v_for_edge="X_0")
    wrong_dag_6.add_edge(u_for_edge="X_0",v_for_edge="Y_0")
    #TODO
    wrong_dag_6.T = 1
    graphs.append(wrong_dag_6)



    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = [ ("Z",), ("X",), ("X", "Z")]
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-5, 5], "Z": [-5.5, 13]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=n_anchor)

    _, _, true_objective_values, _, _, all_CE = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=true_dag,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
        random_state=random_state
    )

    return init_sem, sem, base_dag_view, graphs, exploration_sets, intervention_domain, true_objective_values, all_CE

