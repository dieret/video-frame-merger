# Video Merger

## Requirements

*   OpenCV
*   python3

## Usage

List of command line options:

    python3 merge.py --help
    
## The config file

The file ``configspec.config`` holds a description of all config values 
and their default parameters.

You can supply your own config file to ``merge.py`` by using the
``-c`` (``--config``) option. If you do not specify values for all options, then 
the default values from ``configspec.config`` are used for those options.

You can also change each config value from the command line with the
``-p`` (``--parameter``) option. This option is explained more in the next 
option.

After each run, ``merge.py`` will write out the _effective_ config 
file (with any other parameters already taken into account) to the 
file ``tmp.config``. By default also all unchanged values, comments and 
indents are copied to ``tmp.config`` (if you want to disable that, disable 
the ``r.config.copy_all`` option).

## The -p (--parameter) Option 

Supply one or more option with the ``-p`` (``--parameter``) option:

    ./merge.py -p <key/value pair 1> <key value pair 2>
    
**Important:** Only use ``-p`` once! Do ``-p <key/value pair 1> <key value pair 2>`` 
instead of ``-p <key/value pair 1> -p <key value pair 2>``. 

A key/value pair is specified as:

*   Set value: ``key=value`` 
*   Append value: ``key+=value`` (only for lists!)
*   Remove value: ``key-=value`` (only for lists!)

Here key has the form ``section1.section2.key`` to access a config value
with the key ``key`` in section ``section1`` and subsection ``section2``.

The value is tried to be cast to a float if it contains a ``.`` and 
tried to be cast to an int if not. If the cast fails, it is taken as a
string. Please use ``0`` for ``False`` and 1 for ``True``. 

To specify lists, supply the values separated by ``,``, e.g. 
``key=value1,value2``. 
**Important**: Add a leading or trailing comma, even when only giving one
(or zero) value, e.g. ``key=value1,``, ``key=,``.

Here are some examples:

*   ``-p m.save=metric,``: Save only picutures of the metric
*   ``-p m.save-=final,``: Do not save the final picture
*   ``-p m.image_format=jpg``: Use jpg as image format.

## Examples

### Merging with simple R3 metric

Note: Applying a cutoff is a standard now, so we need to actively specify
the operations applied to the metric to remove it. Here we only apply a
small blur on the (R3) metric.

    ./merge.py data/untracked/rad.mp4 -v metric merge -p m.mpp.operations=gauss,

Alternatively, we could have also just removed ``cutoff`` from the 
operations by using ``-p`` with the ``-=`` syntax.

    ./merge.py data/untracked/rad.mp4 -v metric merge -p m.mpp.operations-=cutoff,


### Cutoff merging

    ./merge.py data/untracked/rad.mp4 -v metric merge m.mpp.cutoff.min=0.0001
 
This is using the default overlay (``m.overlay.strategy=add``). 
Set ``m.mpp.cutoff.min`` to a small number, e.g. ``0.1/number frames``. 
It must be mon zero, because else our background becomes black. 

### Overlay

Note: If we want the overlay to not fade, we need to set ``m/mpp/cutoff/min`` to zero.

    ./merge.py data/untracked/rad.mp4 -v metric merge m.mpp.cutoff.min=0 m.overlay.strategy=overlay 
    
If we want thad fade effect, e.g. run 

    ./merge.py data/untracked/rad.mp4 -v metric merge m.mpp.cutoff.min=0.1 m.overlay.strategy=overlay 
    
Or for a slightly slower fading ``mp.mpp.cutoff.min=0.05``.

### White grid lines

    ./merge.py data/untracked/rad.mp4 -v metric merge -p m.diff.operations=, m.mpp.operations=gauss,cutoff,edge,open,dilate m.mpp.dilate.kernel=2,2  m.mpp.open.kernel=5,5 m.overlay.strategy=overlay m.layer.multiply=100

    

 