import numpy as np
import tensorflow as tf
import sys
import time

def progbar(i, iter_per_epoch, message='', bar_length=50, display=True):
    j = (i % iter_per_epoch) + 1
    end_epoch = j == iter_per_epoch
    if display:
        perc = int(100. * j / iter_per_epoch)
        prog = ''.join(['='] * int(bar_length * perc / 100))
        template = "\r[{:" + str(bar_length) + "s}] {:3d}%. {:s}"
        string = template.format(prog, perc, message)
        sys.stdout.write(string)
        sys.stdout.flush()
        if end_epoch:
            sys.stdout.write('\r{:100s}\r'.format(''))
            sys.stdout.flush()
    return end_epoch, (i + 1)/iter_per_epoch

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    with open('graph.htm','w') as file:
        file.write('<!DOCTYPE html> <html> <body> \n' +
                   iframe +
                   ' </body> </html>')
    print('graph written')

def show_default_graph():
    show_graph(tf.get_default_graph().as_graph_def())
	