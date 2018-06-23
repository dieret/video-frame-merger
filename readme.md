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

You can either supply your own config file to ``merge.py`` by using the
``-c`` (``--config``) option. 

You can also change each config value from the command line with the
``-p`` (``--parameter``) option. 

After each run of ``merge.py`` will write out the _effective_ config 
file (with any other parameters already taken into account) to the 
file ``tmp.config``. 

## Examples

### Merging with simple R3 metric

Note: Applying a cutoff is a standard now, so we need to actively specify
the operations applied to the metric to remove it. Her we only apply a
small blur on the (R3) metric.

    ./merge.py data/untracked/rad.mp4 -v metric merge -p m.mpp.operations=gauss,

### Cutoff merging

    ./merge.py data/untracked/rad.mp4 -v metric merge -p m.mpp.operations=gauss,cutoff m.mpp.cutoff.min=0.0001
 
This is using the default overlay (``m.overlay.strategy=add``). 
Set ``m.mpp.cutoff.min`` to a small number, e.g. ``0.1/number frames``. 
It must be mon zero, because else our background becomes black. 

### Overlay

Note: If we want the overlay to not fade, we need to set ``m/mpp/cutoff/min`` to zero.

    ./merge.py data/untracked/rad.mp4 -v metric merge m.mpp.operations=gauss,cutoff m.mpp.cutoff.min=0 m.overlay.strategy=overlay 
    
If we want thad fade effect, e.g. run 

    ./merge.py data/untracked/rad.mp4 -v metric merge m.mpp.operations=gauss,cutoff m.mpp.cutoff.min=0.1 m.overlay.strategy=overlay 
    
Or for a slightly slower fading ``mp.mpp.cutoff.min=0.05``.

### White grid lines

    ./merge.py data/untracked/rad.mp4 -v metric merge -p m.diff.operations=, m.mpp.operations=gauss,cutoff,edge,open,dilate m.mpp.dilate.kernel=2,2  m.mpp.open.kernel=5,5 m.overlay.strategy=overlay m.layer.multiply=100

    

 