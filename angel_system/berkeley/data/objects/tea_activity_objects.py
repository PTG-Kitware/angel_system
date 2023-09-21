sub_steps = {}
sub_steps['step 1'] = []
sub_steps['step 2'] = []
sub_steps['step 3'] = []
sub_steps['step 4'] = []
sub_steps['step 5'] = []
sub_steps['step 6'] = []
sub_steps['step 7'] = []

###################################################step 1###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('measure-12oz-water')
sub_step.append([['measuring cup', 'water']])
sub_steps['step 1'].append(sub_step)
del sub_step

##############################sub-step 2##############################
sub_step = []
sub_step.append('pour-water-kettle')
sub_step.append([['measuring cup', 'kettle (open)']])
sub_steps['step 1'].append(sub_step)
del sub_step

##############################sub-step 3##############################
#sub_step = []
#sub_step.append('turn-on-kettle')
#sub_step.append([['switch', 'hand']])
#sub_steps['step 1'].append(sub_step)
#del sub_step

###################################################step 2###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('tea-bag-in-mug')
sub_step.append([['mug', 'tea bag'], ['hand', 'tea bag']])
sub_steps['step 2'].append(sub_step)
del sub_step

###################################################step 3###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('thermometer-turn-on')
sub_step.append([['thermometer (open)', 'hand'],
                    ['thermometer (close)', 'hand']])
sub_steps['step 3'].append(sub_step)
del sub_step

##############################sub-step 2##############################
sub_step = []
sub_step.append('thermometer-in-water')
sub_step.append([['thermometer (open)', 'kettle (open)']])
sub_steps['step 3'].append(sub_step)
del sub_step

##############################sub-step 3##############################
sub_step = []
sub_step.append('check-thermometer')
sub_step.append([['thermometer (open)', 'hand']])
sub_steps['step 3'].append(sub_step)
del sub_step

###################################################step 4###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('pour-water-mug')
sub_step.append([['kettle (close)', 'hand'],
                 ['kettle (close)', 'mug + tea bag']])
sub_steps['step 4'].append(sub_step)
del sub_step

###################################################step 5###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('steep')
sub_step.append([['tea bag', 'hand']])
sub_steps['step 5'].append(sub_step)
del sub_step

###################################################step 6###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('discard-tea-bag')
sub_step.append([['tea bag', 'hand']])
sub_steps['step 6'].append(sub_step)
del sub_step

###################################################step 6###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('discard-tea-bag')
sub_step.append([['tea bag', 'hand']])
sub_steps['step 6'].append(sub_step)
del sub_step


###################################################step 7###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('add-honey')
sub_step.append([['jar of honey (open)', 'hand']])
sub_steps['step 7'].append(sub_step)
del sub_step

##############################sub-step 2##############################
sub_step = []
sub_step.append('stir')
sub_step.append([['spoon', 'hand'],
                 ['spoon', 'mug + tea']])
sub_steps['step 7'].append(sub_step)
del sub_step


original_sub_steps = sub_steps


States_Pairs = [
    ['kettle (closed)', 'kettle (open)'],
    ['timer (on)', 'timer (off)'],
    ['thermometer (open)', 'thermometer (close)'],
    ['jar of honey (close)', 'jar of honey (open)'],
    ['mug', 'mug + tea bag', 'mug + water', 'mug + tea bag + water', 'mug + tea']
]

CONTACT_PAIRS = [
    ['measuring cup', 'water'], 
    ['measuring cup', 'kettle (open)'], # step 1
    ['mug', 'tea bag'],
    ['hand', 'tea bag'], # step 2
    ['thermometer (open)', 'hand'],
    ['thermometer (close)', 'hand'],
    ['thermometer (open)', 'kettle (open)'],
    ['thermometer (open)', 'hand'], # step 3
    ['kettle (close)', 'hand'],
    ['kettle (close)', 'mug + tea bag'], # step 4
    ['tea bag', 'hand'], # step 5
    ['tea bag', 'hand'], # step 6
    ['jar of honey (open)', 'hand'],
    ['spoon', 'hand'],
    ['spoon', 'mug + tea'] # step 7
]

contact_pairs_details = CONTACT_PAIRS

R_class = []
allow_repeat_obj = []
