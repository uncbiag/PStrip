nifty_bin = '/usr/local/niftyreg-git/install/bin'

def nifty_reg_bspline(ref, flo, res = False, cpp = False, rmask = False, fmask = False, levels = False ):
    executable = nifty_bin + '/reg_f3d'

    cmd = executable + ' -ref ' + ref + ' -flo ' + flo
    if cpp != False:
        cmd += ' -cpp ' + cpp
    if res != False:
        cmd += ' -res ' + res
    if rmask != False:
        cmd += ' -rmask ' + rmask
    if fmask != False:
        cmd += ' -fmask ' + fmask
    if levels != False:
        cmd += ' -lp ' + str(levels)
    cmd = cmd + ' -sx 10 --lncc 40 -pad 0'
#    cmd = cmd + ' -sx 10 --nmi --rbn 100 --fbn 100 -gpu -pad 0 -pert 1'


    return cmd

def nifty_reg_affine(ref, flo, res = False, aff = False, rmask = False, fmask = False, symmetric = True, init = 'center'):
    executable = nifty_bin + '/reg_aladin' 
    cmd  = executable + ' -ref ' + ref + ' -flo ' + flo
    if res != False: 
        cmd += ' -res ' + res
    if aff != False: 
        cmd += ' -aff ' + aff
    if rmask != False:
        cmd += ' -rmask ' + rmask
    if fmask != False:
        cmd += ' -fmask ' + fmask
    if symmetric == False:
        cmd += ' -noSym'
#    if init != 'center':
#        cmd += ' -' + init
    return cmd

def nifty_reg_transform(ref=False, ref2=False, invAff1 = False, invAff2 = False, invNrr1 = False, invNrr2 = False, invNrr3 = False, disp1 = False, disp2 = False, def1 = False, def2 = False, comp1 = False, comp2 = False, comp3 = False):
    executable = nifty_bin + '/reg_transform'
    cmd = executable
    if ref != False:
        cmd += ' -ref ' + ref
    if ref2 != False:
        cmd += ' -ref2 ' + ref2
    if invAff1 != False and invAff2 != False:
        cmd += ' -invAff ' + invAff1 + ' ' + invAff2
    elif disp1 != False and disp2 != False:
        cmd += ' -disp ' + disp1 + ' ' + disp2
    elif def1 != False and def2 != False:
        cmd += ' -def ' + def1 + ' ' + def2
    elif comp1 != False and comp2 != False and comp3 != False:
        cmd += ' -comp ' + comp1 + ' ' + comp2 + ' ' + comp3
    elif invNrr1 != False and invNrr2 != False and invNrr3 != False:
        cmd += ' -invNrr ' + invNrr1 + ' ' + invNrr2 + ' ' + invNrr3

    return cmd
   
def nifty_reg_resample(ref, flo, trans = False, res = False):
    executable = nifty_bin + '/reg_resample'
    cmd = executable + ' -ref ' + ref + ' -flo ' + flo
    if trans != False:
        cmd += ' -trans ' + trans
    if res != False:
        cmd += ' -res ' + res

    return cmd
