I like the convenience of hosting and editing this directly on github. 

However, there are certain features that it would be nice to have at least slightly automated:
* table of contents
* testing links
* moving/relabeling document sub-trees
* auto-building stubs for things like topics
* auto-building some kind of navigable digraph structure (in particular, being able to navigate from a page to the pages that link to it)

I don't think these things have to be mutually exclusive. 
I'm thinking of maybe adding some kind of in-line annotations that a script understands how to parse, and then runs automation steps as necessary using github actions or jenkins.

This way, I can leverage this simple github serving pattern without completely giving up on some nicer features.

Collapsing sections for TOC (sadly, doesn't indent when nested): 
* https://gist.github.com/pierrejoubert73/902cc94d79424356a8d20be2b382e1ab
* https://github.com/vsch/idea-multimarkdown/issues/341#:~:text=The%20details%20are%20for%20adding%20collapsible%20sections%20in,you%20finally%20process%20it%2C%20like%20GitHub%2C%20for%20example.
