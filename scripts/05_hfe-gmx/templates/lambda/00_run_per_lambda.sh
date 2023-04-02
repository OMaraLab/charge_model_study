# exit script when command fails, don't keep going
set -e

# ======= set name variables =======
# you can manually set these if you are repurposing for own use

LAMBDA_VALUE={{ lambda_value }}
LAMBDA_NAME=lambda-{{ lambda_value }}
CHARGE_MODEL={{ charge_model }}
MOLNAME={{ molecule_name }}
SYSNAME="${MOLNAME}_${CHARGE_MODEL}"

echo "Lambda: ${LAMBDA_VALUE}"
echo "Lambda name: ${LAMBDA_NAME}"
echo "Charge model: ${CHARGE_MODEL}"
echo "Molecule name: ${MOLNAME}"
echo "System name: ${SYSNAME}"

# ======= set general variables =======

TOPFILE="${SYSNAME}.top"
GROMPP="gmx grompp -p ${TOPFILE} -maxwarn 1"
MDRUN="gmx mdrun -nb gpu -v -deffnm"

MDP_DIRECTORY="../../../../../mdps"
LAMBDA_MIN_MDP="${MDP_DIRECTORY}/lambda-min/min-${LAMBDA_NAME}.mdp"
LAMBDA_NVT_MDP="${MDP_DIRECTORY}/lambda-nvt/nvt-${LAMBDA_NAME}.mdp"
LAMBDA_NPT_MDP="${MDP_DIRECTORY}/lambda-npt/npt-${LAMBDA_NAME}.mdp"
LAMBDA_PROD_MDP="${MDP_DIRECTORY}/lambda-prod/prod-${LAMBDA_NAME}.mdp"

# ====== begin gromacs part ======
# ------ setup initial files ------
STEP_05_SOLVMIN_NAME="05_${SYSNAME}_min-solv"
STEP_05_SOLVMIN_GROFILE="${STEP_05_SOLVMIN_NAME}.gro"


# ------ minimize again ------
STEP_06_MIN_NAME="06_${SYSNAME}_min_${LAMBDA_NAME}"
STEP_06_MIN_GROFILE="${STEP_06_MIN_NAME}.gro"
if [ ! -f $STEP_06_MIN_GROFILE ] ; then
    $GROMPP -f $LAMBDA_MIN_MDP -c $STEP_05_SOLVMIN_GROFILE -o ${STEP_06_MIN_NAME}.tpr
    $MDRUN $STEP_06_MIN_NAME
fi

# ------ equilibrate nvt ------
STEP_07_NVT_NAME="07_${SYSNAME}_nvt_${LAMBDA_NAME}"
STEP_07_NVT_GROFILE="${STEP_07_NVT_NAME}.gro"
if [ ! -f $STEP_07_NVT_GROFILE ]; then
    $GROMPP -f $LAMBDA_NVT_MDP -c $STEP_06_MIN_GROFILE -o ${STEP_07_NVT_NAME}.tpr
    $MDRUN $STEP_07_NVT_NAME
fi

# ------ equilibrate npt ------
STEP_08_NPT_NAME="08_${SYSNAME}_npt_${LAMBDA_NAME}"
STEP_08_NPT_GROFILE="${STEP_08_NPT_NAME}.gro"
if [ ! -f $STEP_08_NPT_GROFILE ] ; then
    $GROMPP -f $LAMBDA_NPT_MDP -c $STEP_07_NVT_GROFILE -o ${STEP_08_NPT_NAME}.tpr
    $MDRUN $STEP_08_NPT_NAME
fi

# ------ production run ------
STEP_09_PROD_NAME="09_${SYSNAME}_prod_${LAMBDA_NAME}"
STEP_09_PROD_GROFILE="${STEP_09_PROD_NAME}.gro"
STEP_09_PROD_XVGFILE="${STEP_09_PROD_NAME}.xvg"

if [ ! -f $STEP_09_PROD_GROFILE ]; then
    if [ ! -f ${STEP_09_PROD_NAME}.tpr ]; then
        $GROMPP -f $LAMBDA_PROD_MDP -c $STEP_08_NPT_GROFILE -o ${STEP_09_PROD_NAME}.tpr
    fi
    $MDRUN $STEP_09_PROD_NAME -dhdl $STEP_09_PROD_XVGFILE -cpi ${STEP_09_PROD_NAME}.cpt
fi

# ------ analyze ------
STEP_10_ANALYZE_XVGFILE="10_${SYSNAME}_analyze_${LAMBDA_NAME}.xvg"
if [ ! -f $STEP_10_ANALYZE_XVGFILE ]; then
    gmx analyze -f $STEP_09_PROD_XVGFILE -ee $STEP_10_ANALYZE_XVGFILE
fi

cd ../{{next_lambda_directory}} && ./00_run_per_lambda.sh > run.log 2>&1
