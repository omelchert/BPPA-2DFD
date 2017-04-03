
function singleRun {
    XOFF=$1
    echo "SOLVING INSTANCE AT XOFF= " $XOFF
    time python main_solveInstance_GaussianBeam.py  $XOFF   
}


singleRun 0.0
singleRun 0.5
singleRun 1.0
singleRun 1.5
singleRun 2.0
singleRun 2.5
singleRun 3.0
singleRun 5.0

time python main_solveInstance_guidingMode.py
