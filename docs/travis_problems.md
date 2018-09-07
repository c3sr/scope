If you see something like 

    error: Server does not allow request for unadvertised object 
    
You might have committed changes in a submodule, then added the updated submodule to scope, and pushed scope, without pushing the submodule changes.
