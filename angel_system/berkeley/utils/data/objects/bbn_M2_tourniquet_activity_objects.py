sub_steps = {}
sub_steps['step 1'] = []
sub_steps['step 2'] = []
sub_steps['step 3'] = []
sub_steps['step 4'] = []
sub_steps['step 5'] = []
sub_steps['step 6'] = []
sub_steps['step 7'] = []
sub_steps['step 8'] = []

"""
sub_steps['step 1.5'] = []
sub_steps['step 2.5'] = []
sub_steps['step 3.5'] = []
sub_steps['step 4.5'] = []
sub_steps['step 5.5'] = []
sub_steps['step 6.5'] = []
sub_steps['step 7.5'] = []

sub_steps['before'] = []
sub_steps['after'] = []
"""
###################################################step 1###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Place tourniquet over affected extremity 2-3 inches above wound site.')
sub_step.append([['hand', 'tourniquet_tourniquet'],
                 #['tourniquet_tourniquet', 'casualty_leg']
                 ]) # tourniquet location varies by video!!
sub_steps['step 1'].append(sub_step)
del sub_step

###################################################step 2###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Pull tourniquet tight.')
sub_step.append([['hand', 'tourniquet_tourniquet']])
sub_steps['step 2'].append(sub_step)
del sub_step

###################################################step 3###################################################
##############################sub-step 1##############################
# Only for tourniquet v0.5
# Removed in tourniquet v0.51+
#sub_step = []
#sub_step.append('Cinch tourniquet strap.')
#sub_step.append([['hand', 'tourniquet_tourniquet']])
#sub_steps['step 3'].append(sub_step)
#del sub_step

###################################################step 3###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Apply strap to strap body.')
sub_step.append([['hand', 'tourniquet_tourniquet']])
sub_steps['step 3'].append(sub_step)
del sub_step

###################################################step 4###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Turn windless clock wise or counter clockwise until hemorrhage is controlled.')
sub_step.append([['hand', 'tourniquet_windlass']])
sub_steps['step 4'].append(sub_step)
del sub_step

###################################################step 5###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Lock windless into the windless keeper.')
sub_step.append([['hand', 'tourniquet_windlass']])
sub_steps['step 5'].append(sub_step)
del sub_step

###################################################step 6###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Pull remaining strap over the windless keeper.')
sub_step.append([['hand', 'tourniquet_tourniquet']])
sub_steps['step 6'].append(sub_step)
del sub_step

###################################################step 7###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Secure strap and windless keeper with keeper securing device.')
sub_step.append([['hand', 'tourniquet_label'],
                 ['hand', 'tourniquet_tourniquet']])
sub_steps['step 7'].append(sub_step)
del sub_step

###################################################step 8###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Mark time on securing device strap with permanent marker.')
sub_step.append([['hand', 'tourniquet_pen'], # right or left hand???
                 ['tourniquet_pen', 'tourniquet_label']])
sub_steps['step 8'].append(sub_step)
del sub_step

"""
###################################################step 1.5###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('In between steps 1 and 2.')
sub_step.append([['hand', 'tourniquet_tourniquet']])
sub_steps['step 1.5'].append(sub_step)
del sub_step

###################################################step 2.5###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('In between steps 2 and 3.')
sub_step.append([['hand', 'tourniquet_tourniquet']])
sub_steps['step 2.5'].append(sub_step)
del sub_step

###################################################step 3.5###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('In between steps 3 and 4.')
sub_step.append([['hand', 'tourniquet_tourniquet']])
sub_steps['step 3.5'].append(sub_step)
del sub_step

###################################################step 4.5###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('In between steps 4 and 5.')
sub_step.append([['hand', 'tourniquet_tourniquet']])
sub_steps['step 4.5'].append(sub_step)
del sub_step

###################################################step 5.5###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('In between steps 5 and 6.')
sub_step.append([['hand', 'tourniquet_tourniquet']])
sub_steps['step 5.5'].append(sub_step)
del sub_step

###################################################step 6.5###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('In between steps 6 and 7.')
sub_step.append([['hand', 'tourniquet_tourniquet']])
sub_steps['step 6.5'].append(sub_step)
del sub_step

###################################################step 8.5###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('In between steps 7 and 8.')
sub_step.append([['hand', 'tourniquet_tourniquet']])
sub_steps['step 7.5'].append(sub_step)
del sub_step

###################################################before###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Not started.')
sub_step.append([['hand', 'tourniquet_tourniquet']])
sub_steps['before'].append(sub_step)
del sub_step

###################################################finished###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Finished.')
sub_step.append([['hand', 'tourniquet_tourniquet']])
sub_steps['finished'].append(sub_step)
del sub_step
"""






CONTACT_PAIRS = contact_pairs_details = [
    [['hand', 'tourniquet_tourniquet'],
     #['tourniquet_tourniquet', 'casualty_leg']
    ], # step 1
    [['hand', 'tourniquet_tourniquet']], # step 2
    #[['hand', 'tourniquet_tourniquet']], # step 3
    [['hand', 'tourniquet_tourniquet']], # step 3
    [['hand', 'tourniquet_windlass']], # step 4
    [['hand', 'tourniquet_windlass']], # step 5
    [['hand', 'tourniquet_tourniquet']], # step 6
    [['hand', 'tourniquet_label'],
     ['hand', 'tourniquet_tourniquet']], # step 7
    [['hand', 'tourniquet_pen'],
     ['tourniquet_pen', 'tourniquet_label']] # step 8
]


States_Pairs = []

