# Video Merger

## Requirements

*   ``python3``
*   ``openCV`` (``cv2`` package of ``python3``)
*   ``ffmpeg`` (if you want to burst movies/gifs or animate a series 
    of pictuers)
*   ``pip install opencv-contrib-python`` to open mp4s

## Usage

List of command line options:

```sh
python3 merge.py --help
```

This will look like this (not up to date):

```
usage: merge.py [-h] [-c CONFIG]
            [-i {SingleFramesIterator,VideoFrameIterator,BurstFrameIterator}]
            [-n NAME] [-s SAVE [SAVE ...]] [-v PREVIEW [PREVIEW ...]]
            [-p PARAMETER [PARAMETER ...]]
            input_path [input_path ...]

positional arguments:
  input_path            InputData video file

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to config file.
  -i {SingleFramesIterator,VideoFrameIterator,BurstFrameIterator}, --iterator {SingleFramesIterator,VideoFrameIterator,BurstFrameIterator}
                        Frame iterator. Default: VideoFrameIterator
  -n NAME, --name NAME  Name (will e.g. become output folder name)
  -s SAVE [SAVE ...], --save SAVE [SAVE ...]
                        Which steps to save.
  -v PREVIEW [PREVIEW ...], --preview PREVIEW [PREVIEW ...]
                        Which steps to preview.
  -p PARAMETER [PARAMETER ...], --parameter PARAMETER [PARAMETER ...]
                        Set parameters of your merger. Give strings like
                        <param_name>=<param_value>. To specify subsections,
                        use '.', e.g. 'section1.section2.key'. To give a list
                        of values, make sure param_value contains a ',', even
                        when passing only one value, e.g. 'key=value,'. When
                        passing multiple list members, so key=value1,value2.We
                        will try to convert each value to a float (if it
                        contains a dot) or an int. If both fail, we take it as
                        a string. For lists, you can also write '+=' or '-='
                        to add or remove values from the list.
```

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

    -p <key/value pair 1> <key value pair 2>

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
string. 

**Important** Boolean values: Please use ``0`` for ``False``! Pretty much 
 anything else will evaluate to ``True`` (behaves like python's 
 ``bool()`` cast)!

To specify lists, supply the values separated by ``,``, e.g. 
``key=value1,value2``. 
**Important**: Add a leading or trailing comma, even when only giving one
(or zero) value, e.g. ``key=value1,``, ``key=,``.

Here are some examples:

*   ``-p m.save=metric,``: Save only picutures of the metric
*   ``-p m.save-=final,``: Do not save the final picture
*   ``-p m.image_format=jpg``: Use jpg as image format.

## Examples

All of those examples can be run at once with the ``run_examples.sh`` script.
Note: This will run all lines starting with ``./merge.py -n examples/``, so be 
careful that those also always run.

## Default

```sh
./merge.py -n examples/default data/giflibrary/fencers.gif 

./merge.py data/untracked/rad.mp4 -v metric merge
```

### Merging with simple euclidean metric

Note: Applying a cutoff is a standard now, so we need to actively specify
the operations applied to the metric to remove it. Here we only apply a
small blur on the (euclidean) metric.

```sh
./merge.py -n examples/euclideanmetric data/giflibrary/fencers.gif -v metric merge -p m.mpp.operations=gauss,

./merge.py data/untracked/rad.mp4 -v metric merge -p m.mpp.operations=gauss,
```


### Cutoff merging

```sh
./merge.py -n examples/cutoff data/giflibrary/fencers.gif -v metric merge -p m.mpp.cutoff.min=0.0001

./merge.py data/untracked/rad.mp4 -v metric merge -p m.mpp.cutoff.min=0.0001 
```

This is using the default overlay (``m.overlay.strategy=add``). 
Set ``m.mpp.cutoff.min`` to a small number, e.g. ``0.1/number frames``. 
It must be mon zero, because else our background becomes black. 

### Overlay

Note: If we want the overlay to not fade, we need to set ``m/mpp/cutoff/min`` to zero.

```sh
./merge.py -n examples/overlay data/giflibrary/fencers.gif -v merge -s final merge -p m.mpp.cutoff.min=0 m.overlay.strategy=overlay 
```

If we want thad fade effect, e.g. run 

```sh
./merge.py data/untracked/rad.mp4 -v metric merge -p m.mpp.cutoff.min=0.1 m.overlay.strategy=overlay 

./merge.py -n examples/overlay2 data/giflibrary/fencers.gif -s merge -v metric merge -p m.mpp.cutoff.min=0.1 m.overlay.strategy=overlay 
```

Or for a slightly slower fading ``mp.mpp.cutoff.min=0.05``.

### White grid lines

```sh
./merge.py -n examples/whitegrid data/giflibrary/fencers.gif -v metric merge -p m.diff.operations=, m.mpp.operations=gauss,cutoff,edge m.overlay.strategy=overlay m.layer.multiply=100

./merge.py data/untracked/rad.mp4 -v metric merge -p m.diff.operations=, m.mpp.operations=gauss,cutoff,edge,open,dilate m.mpp.dilate.kernel=2,2  m.mpp.open.kernel=5,5 m.overlay.strategy=overlay m.layer.multiply=100
```
