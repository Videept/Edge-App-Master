# EdgeApp
Command to create a tunnel between oak and lily. Lily listens to port 60006 and forwards the packte is receives to port 5000 in oak. 
```
ssh -fNT -i ssh_victor -R 60006:localhost:5000 victor@lily.scss.tcd.ie
```

alternatively, using socat:

socat tcp-listen:60006,reuseaddr,fork tcp:oak.scss.tcd.ie:5000
