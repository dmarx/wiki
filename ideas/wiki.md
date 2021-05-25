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
