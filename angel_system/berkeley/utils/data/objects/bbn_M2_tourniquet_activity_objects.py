sub_steps = {}
sub_steps['step 1'] = []
sub_steps['step 2'] = []
sub_steps['step 3'] = []
sub_steps['step 4'] = []
sub_steps['step 5'] = []
sub_steps['step 6'] = []

###################################################step 1###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Place tourniquet with over  effected extremity 2-3 inches above wound site.')
sub_step.append([['hands', 'tourniquet_tourniquet'],
                 ['tourniquet_tourniquet', 'casualty_leg']]) # tourniquet location varies by video!!
sub_steps['step 1'].append(sub_step)
del sub_step

###################################################step 2###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Pull tourniquet tight.')
sub_step.append([['left_hand', 'tourniquet_tourniquet']])
sub_steps['step 2'].append(sub_step)
del sub_step

###################################################step 3###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Cinch tourniquet strap.')
sub_step.append([['left_hand', 'velcro_strap']])
sub_steps['step 3'].append(sub_step)
del sub_step

###################################################step 4###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Turn windless clock wise or counter clockwise until hemorrhage is controlled .')
sub_step.append([['hands', 'tourniquet_windlass']])
sub_steps['step 4'].append(sub_step)
del sub_step

###################################################step 5###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Lock windless into the windless keeper.')
sub_step.append([['right_hand', 'tourniquet_windlass']])
sub_steps['step 5'].append(sub_step)
del sub_step

###################################################step 6###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Mark time on securing device strap with permanent marker.')
sub_step.append([['right_hand', 'tourniquet_pen'], # right or left hand???
                 ['tourniquet_pen', 'tourniquet_label']])
sub_steps['step 6'].append(sub_step)
del sub_step


contact_pairs_details = [
    [['hands', 'tourniquet_tourniquet'],
     ['tourniquet_tourniquet', 'casualty_leg']],  # step 1

    [['left_hand', 'tourniquet_tourniquet']],  # step 2

    [['left_hand', 'velcro_strap']], # step 3

    [['hands', 'tourniquet_windlass']],  # step 4

    [['right_hand', 'tourniquet_windlass']], # step 5

    [['right_hand', 'tourniquet_pen'], # right or left hand???
     ['tourniquet_pen', 'tourniquet_label']], # step 6
]


States_Pairs = []

                         
CONTACT_PAIRS = [
    [['hands', 'tourniquet_tourniquet'],
     ['tourniquet_tourniquet', 'casualty_leg']],  # step 1

    [['left_hand', 'tourniquet_tourniquet']],  # step 2

    [['left_hand', 'velcro_strap']], # step 3

    [['hands', 'tourniquet_windlass']],  # step 4

    [['right_hand', 'tourniquet_windlass']], # step 5

    [['right_hand', 'tourniquet_pen'], # right or left hand???
     ['tourniquet_pen', 'tourniquet_label']], # step 6
]
