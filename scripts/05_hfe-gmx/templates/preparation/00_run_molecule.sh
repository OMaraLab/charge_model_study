# exit script when command fails, don't keep going
set -e

# ======= set name variables =======
# you can manually set these if you are repurposing for own use

CHARGE_MODEL="{{ charge_model }}"
MOLNAME="{{ molecule_name }}"
SYSNAME="${MOLNAME}_${CHARGE_MODEL}"

echo "Charge model: ${CHARGE_MODEL}"
echo "Molecule name: ${MOLNAME}"
echo "System name: ${SYSNAME}"

# ======= set general variables =======

MDP_DIRECTORY="../../../../../mdps"

TOPFILE="${SYSNAME}.top"
GROMPP="gmx grompp -p ${TOPFILE} -maxwarn 1"
MDRUN="gmx mdrun -v -deffnm"

MIN_MDP="${MDP_DIRECTORY}/min.mdp"

# ====== begin gromacs part ======
####  GROMACS time!  ####
# ------ setup initial files ------
STEP_01_GROFILE="01_${SYSNAME}.gro"

# ensure you have mol.pdb and mol.top !!
if [ ! -f $TOPFILE ] ; then
    cp mol.top "01_${SYSNAME}.top"
    cp mol.top $TOPFILE
fi
if [ ! -f $STEP_01_GROFILE ] ; then
    cp ../mol.gro $STEP_01_GROFILE
fi

# ------ create box ------
STEP_02_BOXFILE="02_${SYSNAME}_box.gro"
if [ ! -f $STEP_02_BOXFILE ] ; then
    gmx editconf -f $STEP_01_GROFILE -o $STEP_02_BOXFILE -bt dodecahedron -d 1.5
fi

# ------ minimize in vacuum ------
STEP_03_MIN_NAME="03_${SYSNAME}_min-vac"
STEP_03_MIN_GROFILE="${STEP_03_MIN_NAME}.gro"
vacmin_name="03_${sysname}_min-vac"
if [ ! -f $STEP_03_MIN_GROFILE ]; then
    $GROMPP -f $MIN_MDP -c $STEP_02_BOXFILE -o "${STEP_03_MIN_NAME}.tpr"
    $MDRUN $STEP_03_MIN_NAME
fi

# ------ solvate ------
STEP_04_SOLV_NAME="04_${SYSNAME}_solvated"
STEP_04_SOLV_GROFILE="${STEP_04_SOLV_NAME}.gro"
if [ ! -f $STEP_04_SOLV_GROFILE ]; then
    gmx solvate -cp $STEP_03_MIN_GROFILE -o $STEP_04_SOLV_GROFILE -p $TOPFILE
    cp $TOPFILE solv_${TOPFILE}
fi

# ------ minimize in water ------
STEP_05_SOLVMIN_NAME="05_${SYSNAME}_min-solv"
STEP_05_SOLVMIN_GROFILE="${STEP_05_SOLVMIN_NAME}.gro"
if [ ! -f $STEP_05_SOLVMIN_GROFILE ]; then
    $GROMPP -f $MIN_MDP -c $STEP_04_SOLV_GROFILE -o "${STEP_05_SOLVMIN_NAME}.tpr"
    $MDRUN $STEP_05_SOLVMIN_NAME
fi


# ====== set-up lambda calculations ======

for lambda_dir in $(ls -d ../02_lambda*); do
    cp $TOPFILE $STEP_05_SOLVMIN_GROFILE $lambda_dir/
    # cd $lambda_dir && sbatch 00_run_per_lambda.sh && cd -
done

cd ../02_lambda-00  && sbatch 00_run_per_lambda.sh