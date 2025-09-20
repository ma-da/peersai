from transformers import pipeline

sumz = pipeline("summarization")
ret = sumz("""
    Federal Communications Commission Chairman Brendan Carr said Thursday that ABC late-night host Jimmy Kimmel appeared to “mislead” the American public about facts regarding conservative activist Charlie Kirk’s killing in the days leading up to his show’s suspension.

    Carr also told CNBC’s “Squawk on the Street” that “we’re not done yet” with the changes in “the media ecosystem” that are consequences of President Donald Trump’s election last fall.

    ABC on Wednesday night said it was pulling “Jimmy Kimmel Live!” off the air “indefinitely” because of the host’s comments, which linked Kirk’s alleged killer, Tyler Robinson, to Trump’s “Make America Great Again” movement.

    “The issue that arose here, where lots and lots of people were upset, was not a joke,” Carr said Thursday on CNBC.
""")
print(ret)
