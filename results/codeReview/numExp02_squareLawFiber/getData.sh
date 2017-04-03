
function singleRun {
    XOFF=$1
    echo "SOLVING INSTANCE AT XOFF= " $XOFF
    time python main_solveInstance.py  $XOFF   
}


singleRun 0.0
singleRun 5.0
singleRun 10.0

