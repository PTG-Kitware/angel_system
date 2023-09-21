sub_steps = {}
sub_steps['step 1'] = []
sub_steps['step 2'] = []
sub_steps['step 3'] = []
sub_steps['step 4'] = []
sub_steps['step 5'] = []
sub_steps['step 6'] = []
sub_steps['step 7'] = []
sub_steps['step 8'] = []

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
sub_step = []
sub_step.append('turn-on-kettle')
sub_step.append([['switch', 'hand']])
sub_steps['step 1'].append(sub_step)
del sub_step

###################################################step 2###################################################
##############################sub-step 4##############################
sub_step = []
sub_step.append('place-dipper-on-mug')
sub_step.append([['hand', 'filter cone'],
                    ['hand', 'mug'],
                    ['mug', 'filter cone'],
                    ['hand', 'filter cone + mug']])
sub_steps['step 2'].append(sub_step)
del sub_step

###################################################step 3###################################################
#############################sub-step 5##############################
sub_step = []
sub_step.append('filter-fold-half')
sub_step.append([['paper filter', 'paper filter bag'],
    ['hand', 'paper filter bag'],
    ['hand', 'paper filter'],
    ['hand', 'paper filter (semi)']])
sub_steps['step 3'].append(sub_step)
del sub_step

##############################sub-step 6##############################
sub_step = []
sub_step.append('filter-fold-quarter')
sub_step.append([['hand', 'paper filter (quarter)']])
sub_steps['step 3'].append(sub_step)
del sub_step

# #############################sub-step 7##############################
sub_step = []
sub_step.append('place-filter')
sub_step.append([['paper filter (quarter)', 'filter cone + mug']])
sub_steps['step 3'].append(sub_step)
del sub_step

##############################sub-step 8##############################
sub_step = []
sub_step.append('spread-filter')
sub_step.append([['hand', 'paper filter + filter cone + mug']])
sub_steps['step 3'].append(sub_step)
del sub_step

###################################################step 4###################################################
##############################sub-step 9##############################
sub_step = []
sub_step.append('scale-turn-on')
sub_step.append([['hand', 'scale (off)'],
                    ['hand', 'scale (on)']])
sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 10##############################
sub_step = []
sub_step.append('place-bowl-on-scale')
sub_step.append([['container', 'scale (off)'],
                    ['container', 'scale (on)']])
sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 11##############################
sub_step = []
sub_step.append('zero-scale')
sub_step.append([['hand', 'container + scale']])
sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 12##############################
sub_step = []
sub_step.append('measure-coffee-beans')
sub_step.append([['container + scale', 'coffee bag'],
                    ['coffee beans + container + scale', 'coffee bag'],
                    ['hand', 'coffee bag']])
sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 13##############################
sub_step = []
sub_step.append('pour-coffee-grinder')
sub_step.append([['coffee beans + container', 'grinder (open)'],
                    ['container', 'grinder (open)']])
sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 14##############################
"""
sub_step = []
sub_step.append('Use timer')
sub_step.append([['hand', 'timer (else)'],
                    ['hand', 'timer (20)'],
                    ['hand', 'timer (30)']])
sub_steps['step 4'].append(sub_step)
del sub_step
"""

# ##############################sub-step 15##############################
sub_step = []
sub_step.append('grind-beans')
sub_step.append([['hand', 'grinder (close)']])
sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 16##############################
sub_step = []
sub_step.append('pour-beans-filter')
sub_step.append([['paper filter + filter cone + mug', 'grinder (open)'],
                    ['paper filter + filter cone', 'grinder (open)'],
                    ['coffee grounds + paper filter + filter cone + mug', 'grinder (open)'],
                    ['coffee grounds + paper filter + filter cone', 'grinder (open)']])
sub_steps['step 4'].append(sub_step)
del sub_step

###################################################step 5###################################################
##############################sub-step 17##############################
sub_step = []
sub_step.append('thermometer-turn-on')
sub_step.append([['thermometer (open)', 'hand'],
                    ['thermometer (close)', 'hand']])
sub_steps['step 5'].append(sub_step)
del sub_step

##############################sub-step 18##############################
sub_step = []
sub_step.append('thermometer-in-water')
sub_step.append([['thermometer (open)', 'kettle (open)']])
sub_steps['step 5'].append(sub_step)
del sub_step

###################################################step 6###################################################
##############################sub-step 18##############################
# sub_step = []
# sub_step.append('Set timer to 30 seconds.')
# sub_step.append([['hand', 'timer'],
#                  ['hand', 'timer (20)'],
#                  ['hand', 'timer (0)'],
#                  ['hand', 'timer (30)']])
# sub_steps['step 6'].append(sub_step)
# del sub_step

##############################sub-step 19##############################
sub_step = []
sub_step.append('pour-water-grounds-wet')
sub_step.append([['kettle', 'coffee grounds + paper filter + filter cone + mug']])
sub_steps['step 6'].append(sub_step)
del sub_step

##############################sub-step 20##############################
# sub_step = []
# sub_step.append('Wait about 30 seconds.')
# sub_step.append([
#     # ['hand', 'timer'],
#     #              ['hand', 'timer (20)'],
#     #              ['hand', 'timer (0)'],
#     #              ['hand', 'timer (30)']
# ])
# sub_steps['step 6'].append(sub_step)
# del sub_step

###################################################step 7###################################################
#############################sub-step 20##############################
sub_step = []
sub_step.append('pour-water-grounds-circular')
sub_step.append([['kettle', 'water + coffee grounds + paper filter + filter cone + mug']])
sub_steps['step 7'].append(sub_step)
del sub_step

##################################################step 8###################################################
#############################sub-step 21##############################
sub_step = []
sub_step.append('remove-dripper')
sub_step.append([['hand', 'used paper filter + filter cone'],
                    ['hand', 'used paper filter + filter cone + mug']])
sub_steps['step 8'].append(sub_step)

##############################sub-step 22##############################
sub_step = []
sub_step.append('remove-grounds')
sub_step.append([
                # ['hand', 'used paper filter + filter cone'],
                    # ['filter cone', 'used paper filter'],
                    ['hand', 'used paper filter']])
sub_steps['step 8'].append(sub_step)

#############################sub-step 23##############################
sub_step = []
sub_step.append('discard-grounds')
sub_step.append([['used paper filter', 'trash can']
                    # , ['hand', 'used paper filter']
                    ])
sub_steps['step 8'].append(sub_step)


# This version of the sub-steps allows us to make 1:1 matches with the activity labels
original_sub_steps = {}
original_sub_steps['step 1'] = []
original_sub_steps['step 2'] = []
original_sub_steps['step 3'] = []
original_sub_steps['step 4'] = []
original_sub_steps['step 5'] = []
original_sub_steps['step 6'] = []
original_sub_steps['step 7'] = []
original_sub_steps['step 8'] = []

###################################################step 1###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('measure-12oz-water')
sub_step.append([['measuring cup', 'water']])
original_sub_steps['step 1'].append(sub_step)
del sub_step

##############################sub-step 2##############################
sub_step = []
sub_step.append('pour-water-kettle')
sub_step.append([['measuring cup', 'kettle (open)']])
original_sub_steps['step 1'].append(sub_step)
del sub_step

##############################sub-step 3##############################
sub_step = []
sub_step.append('turn-on-kettle')
sub_step.append([['switch', 'hand']])
original_sub_steps['step 1'].append(sub_step)
del sub_step

###################################################step 2###################################################
##############################sub-step 4##############################
sub_step = []
sub_step.append('place-dipper-on-mug')
sub_step.append([['hand', 'filter cone'],
                    ['hand', 'mug'],
                    ['mug', 'filter cone'],
                    ['hand', 'filter cone + mug']])
original_sub_steps['step 2'].append(sub_step)
del sub_step

###################################################step 3###################################################
#############################sub-step 5##############################
sub_step = []
sub_step.append('filter-fold-half')
sub_step.append([['paper filter', 'paper filter bag'],
    ['hand', 'paper filter bag'],
    ['hand', 'paper filter'],
    ['hand', 'paper filter (semi)']])
original_sub_steps['step 3'].append(sub_step)
del sub_step

##############################sub-step 6##############################
sub_step = []
sub_step.append('filter-fold-quarter')
sub_step.append([['hand', 'paper filter (quarter)']])
original_sub_steps['step 3'].append(sub_step)
del sub_step

# #############################sub-step 7##############################
sub_step = []
sub_step.append('place-filter')
sub_step.append([['hand', 'paper filter (quarter)'],
                 ['paper filter (quarter)', 'filter cone + mug']])
original_sub_steps['step 3'].append(sub_step)
del sub_step

##############################sub-step 8##############################
sub_step = []
sub_step.append('spread-filter')
sub_step.append([['hand', 'paper filter + filter cone + mug']])
original_sub_steps['step 3'].append(sub_step)
del sub_step

###################################################step 4###################################################
##############################sub-step 9##############################
sub_step = []
sub_step.append('scale-turn-on')
sub_step.append([['hand', 'scale (off)'],
                    ['hand', 'scale (on)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 10##############################
sub_step = []
sub_step.append('place-bowl-on-scale')
sub_step.append([['container', 'scale (off)'],
                 ['container', 'scale (on)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 11##############################
sub_step = []
sub_step.append('zero-scale')
sub_step.append([['hand', 'container + scale']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 12##############################
sub_step = []
sub_step.append('measure-coffee-beans')
sub_step.append([['container + scale', 'coffee bag'],
                    ['coffee beans + container + scale', 'coffee bag'],
                    ['hand', 'coffee bag']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 13##############################
sub_step = []
sub_step.append('pour-coffee-grinder')
sub_step.append([['coffee beans + container', 'grinder (open)'],
                    ['container', 'grinder (open)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 14##############################
"""
sub_step = []
sub_step.append('Set timer for 20 seconds')
sub_step.append([['hand', 'timer'],
                 ['hand', 'timer (20)'],
                 ['hand', 'timer (else)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

sub_step = []
sub_step.append('Set timer to 30 seconds')
sub_step.append([['hand', 'timer'],
                 ['hand', 'timer (else)'],
                 ['hand', 'timer (30)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

sub_step = []
sub_step.append('Turn on the timer')
sub_step.append([['hand', 'timer'],
                 ['hand', 'timer (else)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step
"""

# ##############################sub-step 15##############################
sub_step = []
sub_step.append('grind-beans')
sub_step.append([['hand', 'grinder (close)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 16##############################
sub_step = []
sub_step.append('pour-beans-filter')
sub_step.append([['paper filter + filter cone + mug', 'grinder (open)'],
                    ['paper filter + filter cone', 'grinder (open)'],
                    ['coffee grounds + paper filter + filter cone + mug', 'grinder (open)'],
                    ['coffee grounds + paper filter + filter cone', 'grinder (open)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

###################################################step 5###################################################
##############################sub-step 17##############################
sub_step = []
sub_step.append('thermometer-turn-on')
sub_step.append([['thermometer (open)', 'hand'],
                    ['thermometer (close)', 'hand']])
original_sub_steps['step 5'].append(sub_step)
del sub_step

##############################sub-step 18##############################
sub_step = []
sub_step.append('thermometer-in-water')
sub_step.append([['thermometer (open)', 'kettle (open)']])
original_sub_steps['step 5'].append(sub_step)
del sub_step

###################################################step 6###################################################
##############################sub-step 18##############################

##############################sub-step 19##############################
sub_step = []
sub_step.append('pour-water-grounds-wet')
sub_step.append([['kettle', 'coffee grounds + paper filter + filter cone + mug']])
original_sub_steps['step 6'].append(sub_step)
del sub_step

###################################################step 7###################################################
#############################sub-step 20##############################
sub_step = []
sub_step.append('pour-water-grounds-circular')
sub_step.append([['kettle', 'water + coffee grounds + paper filter + filter cone + mug']])
original_sub_steps['step 7'].append(sub_step)
del sub_step

# ???? Allow the rest of the water in the dripper to drain
sub_step = []
sub_step.append('water-drain')
sub_step.append([['used paper filter + filter cone + mug']])

##################################################step 8###################################################
#############################sub-step 21##############################
sub_step = []
sub_step.append('remove-dripper')
sub_step.append([['hand', 'used paper filter + filter cone'],
                    ['hand', 'used paper filter + filter cone + mug']])
original_sub_steps['step 8'].append(sub_step)

##############################sub-step 22##############################
sub_step = []
sub_step.append('remove-grounds')
sub_step.append([['hand', 'used paper filter + filter cone'],
                 ['hand', 'used paper filter']])
original_sub_steps['step 8'].append(sub_step)

#############################sub-step 23##############################
sub_step = []
sub_step.append('discard-grounds')
sub_step.append([['used paper filter', 'trash can'],
                 ['hand', 'used paper filter']
                    ])
original_sub_steps['step 8'].append(sub_step)



States_Pairs = [
    ['kettle (closed)',
        'kettle (open)'],
    ['coffee beans + container', 'coffee beans + container + scale'],
    ['coffee grounds + paper filter + filter cone',
        'coffee grounds + paper filter + filter cone + mug',
        'filter cone', 'filter cone + mug', 'paper filter + filter cone',
        'paper filter + filter cone + mug', 'used paper filter + filter cone',
        'used paper filter + filter cone + mug', 'water + coffee grounds + paper filter + filter cone + mug'],
    ['coffee + mug',
        'coffee grounds + paper filter + filter cone + mug',
        'filter cone + mug', 'mug', 'paper filter + filter cone + mug',
        'used paper filter + filter cone + mug', 'water + coffee grounds + paper filter + filter cone + mug'],
    ['container', 'container + scale'],
    ['scale (off)', 'scale (on)', 'container + scale', 'coffee beans + container + scale'],
    ['paper filter (semi)', 'paper filter (quarter)', 'paper filter'],
    ['coffee beans + container', 'coffee beans + container + scale'],
    ['timer (on)', 'timer (off)'],
    ['thermometer (open)', 'thermometer (close)'],
    ['grinder (close)', 'grinder (open)']
]
    
CONTACT_PAIRS = [
    ['measuring cup', 'water'],
    ['kettle (open)', 'measuring cup'], # step 1

    ['mug', 'filter cone'],  # step 2

    ['paper filter', 'paper filter bag'],
    ['paper filter (semi)', 'filter cone + mug'],
    ['paper filter (quarter)', 'filter cone + mug'],
    ['paper filter', 'filter cone + mug'],  # step 3

    ['scale (on)', 'container'],
    ['scale (off)', 'container'],
    ['container + scale', 'coffee bag'],
    ['coffee beans + container + scale', 'coffee bag'],
    ['coffee beans + container', 'grinder (open)'],
    ['container', 'grinder (open)'],
    ['paper filter + filter cone + mug', 'grinder (open)'],
    ['paper filter + filter cone', 'grinder (open)'],
    ['coffee grounds + paper filter + filter cone', 'grinder (open)'],
    ['coffee grounds + paper filter + filter cone + mug', 'grinder (open)'],  # step 4

    ['thermometer (open)', 'kettle (open)'],
    # ['thermometer', 'kettle'], # step 5

    ['kettle', 'coffee grounds + paper filter + filter cone + mug'],
    ['kettle', 'water + coffee grounds + paper filter + filter cone + mug'],  # step 6 ~ 7

    ['mug', 'used paper filter + filter cone'],
    # ['hand', 'used paper filter + filter cone'],
    # ['hand', 'used paper filter + filter cone + mug'],
    ['used paper filter', 'trash can'],
    ['trash can', 'filter cone'],
    ['hand', 'used paper filter']  # step 8
]

contact_pairs_details = CONTACT_PAIRS

R_class = []
allow_repeat_obj = []
