from graphviz import Digraph

dot = Digraph(comment='GRUModel Architecture', format='png')
dot.attr(rankdir='LR', size='8,5')
dot.node('A', 'Input\n[batch, 168, 4 or 5]', shape='box', style='filled', fillcolor='lightblue')
dot.node('B', 'GRU\n(2 layers, hidden=82)', shape='box', style='filled', fillcolor='lightgray')
dot.node('C', 'Last Time Step\n[batch, 82]', shape='ellipse')
dot.node('D', 'Linear\n(82 → 24)', shape='box', style='filled', fillcolor='lightgray')
dot.node('E', 'Output\n[batch, 24]', shape='box', style='filled', fillcolor='lightgreen')
dot.edges(['AB', 'BC', 'CD', 'DE'])
dot.render('gru_model_architecture', cleanup=True)