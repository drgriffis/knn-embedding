#!/bin/bash
#
# Initial setup for running k-NN calculation
#
# Downloads dependencies as needed

if [ ! -d dependencies ]; then
    mkdir dependencies
fi


#######################################################
### Dependencies/Python configuration #################

PYCONFIG=dependencies/pyconfig.sh

if [ -e ${PYCONFIG} ]; then
    echo "Python configuration file ${PYCONFIG} exists."
    echo "Running git pull on dependencies..."
    if [ -d dependencies/pyemblib ]; then
        cd dependencies/pyemblib
        git pull
        cd ../../
    fi
    if [ -d dependencies/configlogger ]; then
        cd dependencies/configlogger
        git pull
        cd ../../
    fi
    if [ -d dependencies/miscutils ]; then
        cd dependencies/miscutils
        git pull
        cd ../../
    fi
    source ${PYCONFIG}
else
    # start Python configuration file
    echo '#!/bin/bash' > ${PYCONFIG}

    # configure Python installation to use
    echo "Python environment to execute with (should include tensorflow)"
    read -p "Path to binary [default: python3]: " PY
    if [ -z "${PY}" ]; then
        PY=python3
    fi
    echo "export PY=${PY}" >> ${PYCONFIG}

    # check for pyemblib
    ${PY} -c "import pyemblib" 2>>/dev/null
    if [[ $? = 1 ]]; then
        echo
        echo "Cloning pyemblib..."
        cd dependencies
        git clone https://github.com/drgriffis/pyemblib.git
        cd ../
        echo "export PYTHONPATH=\${PYTHONPATH}:$(pwd)/dependencies/pyemblib" >> ${PYCONFIG}
    fi

    # check for configlogger
    ${PY} -c "import configlogger" 2>>/dev/null
    if [[ $? = 1 ]]; then
        echo
        echo "Cloning configlogger"
        cd dependencies
        git clone https://github.com/drgriffis/configlogger.git
        cd ../
        echo "export PYTHONPATH=\${PYTHONPATH}:$(pwd)/dependencies/configlogger" >> ${PYCONFIG}
    fi

    # check for drgriffis.common
    ${PY} -c "import drgriffis.common" 2>>/dev/null
    if [[ $? = 1 ]]; then
        echo
        echo "Cloning miscutils (drgriffis.common)"
        cd dependencies
        git clone https://github.com/drgriffis/miscutils.git
        cd ../
        echo "export PYTHONPATH=\${PYTHONPATH}:$(pwd)/dependencies/miscutils/py" >> ${PYCONFIG}
    fi

    echo
    echo "Python configuration complete."
    echo "Configuration written to ${PYCONFIG}"
fi
