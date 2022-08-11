"""
Provides some insight into the RlCitoStation model by printing out the
contents of its (default) Context.
"""

from pydrake.examples import RlCitoStation


def main():
    station = RlCitoStation()
    station.SetupCitoRlStation()
    station.Finalize()

    context = station.CreateDefaultContext()
    print(context)


if __name__ == '__main__':
    main()
